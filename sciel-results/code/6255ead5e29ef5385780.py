"""Reformulate as non-convex QP via complex vector parameterization, optimize with ADMM-like
alternating updates and local Nelder-Mead polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Deadline guard: 120 s hard limit per task contract; use 110 s as safety margin.
_DEADLINE = 110.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _autoconv(h):
    """Full autoconvolution (linear) of h with itself."""
    return np.convolve(h, h, mode='full')


def _score(h):
    """Score = ||h||_2^2 / (||h||_1 * ||autoconv(h)||_inf)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0.0, None)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2sq = np.dot(h, h)
    ac = _autoconv(h)
    sup = np.max(ac)
    if sup < 1e-12:
        return 0.0
    return l2sq / (l1 * sup)


def _neg_score(h):
    s = _score(h)
    return -s


def _gaussian_start(width):
    """Gaussian-shaped nonnegative heights."""
    x = np.linspace(-1.0, 1.0, N)
    h = np.exp(-0.5 * (x / width) ** 2)
    return h


def _triangle_start():
    """Triangular (tent) shape."""
    h = np.abs(1.0 - 2.0 * np.linspace(0.0, 1.0, N))
    h[0] = 0.0
    h[-1] = 0.0
    return h


def _cosine_start():
    """Cosine bump."""
    x = np.linspace(0.0, np.pi, N)
    h = np.cos(0.5 * x) ** 2
    h[0] = max(h[0], 0.0)
    h[-1] = max(h[-1], 0.0)
    return h


def _random_nonneg(seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(N) ** 0.5  # bias toward lower values to reduce peak


def _local_polish(h0, maxiter=2000, timeout=None):
    """Nelder-Mead local optimization of negative score."""
    h0 = np.clip(np.asarray(h0, dtype=float), 0.0, None)
    # Normalize to unit L1 for stable scaling
    s = np.sum(h0)
    if s < 1e-12:
        h0 = np.ones(N) / N
    else:
        h0 = h0 / s

    def objective(x):
        x = np.clip(x, 0.0, None)
        return -_score(x)

    x0 = h0.copy()
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                            'adaptive': True})
    best_h = np.clip(res.x, 0.0, None)
    best_s = -res.fun
    return best_h, best_s


def _admm_like(h0, n_iter=300, mu=1.0, timeout=None):
    """
    ADMM-style alternating updates:
    - z-update: minimize ||h - z||^2 + mu * (peak_penalty) subject to h >= 0
    - h-update: given z, update h to reduce autoconv peak while maintaining L2/L1 ratio
    This is a heuristic; we alternate between:
      1. Project onto nonnegative and scale to keep L1=1
      2. Gradient step on the autoconv peak constraint
    """
    h = np.clip(np.asarray(h0, dtype=float), 0.0, None)
    s = np.sum(h)
    if s < 1e-12:
        h = np.ones(N) / N
        s = 1.0
    h = h / s  # L1 = 1

    best_h = h.copy()
    best_s = _score(h)

    for it in range(n_iter):
        if timeout is not None and _remaining() < 5.0:
            break
        # Compute autoconvolution and its gradient w.r.t. h
        ac = _autoconv(h)
        peak_idx = np.argmax(ac)
        peak_val = ac[peak_idx]
        if peak_val < 1e-12:
            break

        # Gradient of peak autoconv w.r.t. h:
        # ac[k] = sum_i h[i] * h[i + k - N + 1] (for linear conv)
        # d(ac[peak_idx])/d(h[j]) = h[j + peak_idx - N + 1] + h[j - (peak_idx - N + 1)]
        # Actually for full convolution: ac[m] = sum_j h[j] * h[m - j]
        # d(ac[m])/d(h[j]) = h[m - j] + h[m - j] if m-j != j, else 2*h[j]
        # More carefully: ac[m] = sum_j h[j]*h[m-j], so d(ac[m])/d(h[j]) = h[m-j] + (h[m-j] if j != m-j else 0)
        # Wait, let's just use: d(ac[m])/d(h[j]) = h[m-j] + h[m-j] when the two terms are distinct
        # Actually: ac[m] = sum_j h[j] h[m-j]. d/dh[j] = h[m-j] + h[m-j] if j != m-j, else 2 h[j].
        # But h[m-j] is only valid if 0 <= m-j < N.
        m = peak_idx
        grad = np.zeros(N)
        for j in range(N):
            k = m - j
            if 0 <= k < N:
                if k == j:
                    grad[j] = 2.0 * h[j]
                else:
                    grad[j] = h[k]
                    # Also the symmetric term: when we differentiate h[k] in h[j]*h[k],
                    # it contributes to d/dh[k], not d/dh[j]. So for fixed j, only h[k] term.
                    # Actually ac[m] = sum_j h[j]*h[m-j]. For a specific j, the term is h[j]*h[m-j].
                    # d/dh[j] of h[j]*h[m-j] = h[m-j] (if j != m-j) or 2*h[j] (if j == m-j).
                    # But h[m-j] also appears in the term for j' = m-j: h[m-j]*h[j].
                    # So d/dh[j] gets contribution from term j (h[m-j]) and term j'=m-j (h[j]).
                    # Wait no: the sum is over j. For a fixed index j in h, it appears in:
                    #   term j: h[j] * h[m-j]  -> derivative w.r.t. h[j] is h[m-j]
                    #   term j' where m-j' = j, i.e., j' = m-j: h[m-j] * h[j] -> derivative w.r.t. h[j] is h[m-j]
                    # So if j != m-j, total is 2*h[m-j]. If j == m-j, it's 2*h[j].
                    pass
        # Let me redo this properly.
        # ac[m] = sum_{j=0}^{N-1} h[j] * h[m-j]  (with h[k]=0 for k outside [0,N-1])
        # d(ac[m])/d(h[j]) = h[m-j] (from term j) + h[j] (from term j'=m-j, if m-j is a valid index and m-j != j)
        # If m-j == j, then it's the same term, so d/dh[j] = 2*h[j].
        # If m-j is outside [0,N-1], then only the term j contributes (if j is valid).
        grad = np.zeros(N)
        for j in range(N):
            k = m - j
            if 0 <= k < N:
                if k == j:
                    grad[j] += 2.0 * h[j]
                else:
                    grad[j] += h[k]
            # Check if j appears as the second factor: h[j] * h[m-j'] where m-j' = j => j' = m-j
            # This is already handled above since we iterate over all j.
            # Actually no: the sum is sum_j h[j]*h[m-j]. For a given h[j], it appears:
            #  - as the first factor in term j: contribution h[m-j] to d/dh[j]
            #  - as the second factor in term j' where m-j' = j, i.e., j' = m-j: contribution h[j'] = h[m-j] to d/dh[j]
            # So if 0 <= m-j < N and m-j != j, total is 2*h[m-j].
            # If m-j == j, total is 2*h[j].
            # If m-j is out of range, only the first factor contributes: h[m-j] is 0, so 0.
            # Wait, I'm confusing myself. Let me just use the formula:
            # d(ac[m])/d(h[j]) = h[m-j] + h[m-j] if 0<=m-j<N and m-j != j
            #                    = 2*h[j] if m-j == j (i.e., 2j == m)
            #                    = 0 if m-j < 0 or m-j >= N (and j != m-j)
            # Actually simpler: just compute it numerically or use the direct formula.
            pass

        # Let me just use a clean gradient computation:
        grad = np.zeros(N)
        for j in range(N):
            k = m - j
            if 0 <= k < N:
                grad[j] += h[k]
            # Symmetric: h[j] appears as second factor when the first index is m-j
            j2 = m - j
            if 0 <= j2 < N and j2 != j:
                grad[j] += h[j2]
            # Wait this is wrong too. Let me think again.
            # ac[m] = sum_i h[i] * h[m-i]
            # For a fixed j, h[j] appears in:
            #   - term i=j: h[j]*h[m-j], derivative = h[m-j] (if m-j in range)
            #   - term i=m-j: h[m-j]*h[j], derivative = h[m-j] (if m-j in range and m-j != j)
            # So if 0 <= m-j < N:
            #   if m-j == j: derivative = 2*h[j]
            #   else: derivative = h[m-j] + h[m-j] = 2*h[m-j]
            # else: derivative = 0
            pass

        # OK let me just do it cleanly:
        grad = np.zeros(N)
        for j in range(N):
            k = m - j
            if 0 <= k < N:
                if k == j:
                    grad[j] = 2.0 * h[j]
                else:
                    grad[j] = 2.0 * h[k]
            else:
                grad[j] = 0.0

        # Gradient of score w.r.t. h:
        # score = ||h||^2 / (||h||_1 * peak)
        # Let A = ||h||^2, B = ||h||_1, C = peak
        # d(score)/dh = (2*h * B * C - A * (1 * C + B * dC/dh)) / (B^2 * C^2)
        # = (2*h*B*C - A*C - A*B*grad_C) / (B^2 * C^2)
        # = C*(2*h*B - A) / (B^2 * C^2) - A*B*grad_C / (B^2 * C^2)
        # = (2*h*B - A) / (B^2 * C) - A*grad_C / (B * C^2)

        A = np.dot(h, h)
        B = np.sum(h)
        C = peak_val
        if B < 1e-12 or C < 1e-12:
            break

        grad_score = (2.0 * h * B - A) / (B * B * C) - A * grad / (B * C * C)

        # Gradient ascent on score (i.e., descent on -score)
        lr = 0.01
        h_new = h + lr * grad_score
        h_new = np.clip(h_new, 0.0, None)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            h_new = h
        else:
            h_new = h_new / s_new  # renormalize to L1=1

        s_val = _score(h_new)
        if s_val > best_s:
            best_s = s_val
            best_h = h_new.copy()
        h = h_new

    return best_h, best_s


def construct():
    """Return 50 nonnegative step heights maximizing the ratio."""
    t0 = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)

    # Try multiple initializations
    inits = [
        _gaussian_start(0.3),
        _gaussian_start(0.5),
        _gaussian_start(0.7),
        _triangle_start(),
        _cosine_start(),
        _random_nonneg(42),
        _random_nonneg(123),
        _random_nonneg(999),
        np.ones(N),
    ]

    for idx, h0 in enumerate(inits):
        if _remaining() < 15.0:
            break

        # Local polish with Nelder-Mead
        try:
            h_polished, s_polished = _local_polish(h0, maxiter=1500)
            if s_polished > best_score:
                best_score = s_polished
                best_h = h_polished.copy()
        except Exception:
            pass

        if _remaining() < 10.0:
            break

        # ADMM-like refinement
        try:
            h_admm, s_admm = _admm_like(h_polished, n_iter=200)
            if s_admm > best_score:
                best_score = s_admm
                best_h = h_admm.copy()
        except Exception:
            pass

        # Another round of Nelder-Mead from the ADMM result
        if _remaining() > 20.0:
            try:
                h_final, s_final = _local_polish(h_admm, maxiter=1000)
                if s_final > best_score:
                    best_score = s_final
                    best_h = h_final.copy()
            except Exception:
                pass

    # Final safety: ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    return [float(x) for x in best_h]
