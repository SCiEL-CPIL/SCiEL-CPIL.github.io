"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Uses a smooth-max (log-sum-exp) formulation with spectral parameterization,
projected gradient descent with Nesterov momentum, and SLSQP polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100


def _linear_overlap_values(h):
    """Compute overlap O_k for all shifts k=1..N-1.
    O_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    """
    n = len(h)
    overlaps = np.empty(n - 1)
    for k in range(1, n):
        overlaps[k - 1] = np.dot(h[:n - k], 1.0 - h[k:])
    return overlaps


def _score(h):
    """Return the maximum overlap (the quantity to minimize)."""
    return float(np.max(_linear_overlap_values(h)))


def _smooth_max_value(v, t):
    """Smooth max via log-sum-exp: (1/t) log(sum exp(t*v))."""
    v = np.asarray(v, dtype=float)
    vmax = np.max(v)
    return float(vmax + np.log(np.sum(np.exp(t * (v - vmax)))) / t)


def _smooth_max_grad(v, t):
    """Gradient of smooth max w.r.t. v."""
    v = np.asarray(v, dtype=float)
    vmax = np.max(v)
    w = np.exp(t * (v - vmax))
    w /= np.sum(w)
    return w


def _fix_sum(h):
    """Adjust h so that sum(h) == N/2, keeping h in [0,1]."""
    target = N / 2.0
    h = np.asarray(h, dtype=float)
    diff = target - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    residual = target - np.sum(h)
    if abs(residual) > 1e-12:
        if residual > 0:
            idx = np.argmin(1.0 - h)
        else:
            idx = np.argmax(h)
        h[idx] = np.clip(h[idx] + residual, 0.0, 1.0)
    return h


def _spectral_h_from_coeffs(coeffs, n_modes):
    """Convert spectral coefficients to h in [0,1] with mean 0.5."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    for m in range(1, n_modes + 1):
        a = coeffs[2 * m - 2]
        b = coeffs[2 * m - 1]
        h += a * np.cos(2 * np.pi * m * i / N) + b * np.sin(2 * np.pi * m * i / N)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _h_to_spectral_coeffs(h, n_modes):
    """Extract spectral coefficients from h."""
    H = np.fft.fft(h)
    coeffs = np.zeros(2 * n_modes)
    for m in range(1, n_modes + 1):
        coeffs[2 * m - 2] = 2.0 * np.real(H[m]) / N
        coeffs[2 * m - 1] = -2.0 * np.imag(H[m]) / N
    return coeffs


def _objective_and_grad(coeffs, n_modes, t):
    """Return (value, grad) of smooth-max overlap w.r.t. spectral coeffs.

    We compute the gradient via the chain rule:
      d(smooth_max)/d(coeffs) = sum_k w_k * d(O_k)/d(coeffs)
    where w_k are the softmax weights.
    """
    h = _spectral_h_from_coeffs(coeffs, n_modes)
    ov = _linear_overlap_values(h)
    val = _smooth_max_value(ov, t)
    w = _smooth_max_grad(ov, t)  # shape (N-1,)

    # Gradient of O_k w.r.t. h: for shift k, O_k = sum_{i=0}^{N-1-k} h[i](1-h[i+k])
    # dO_k/dh[i] = (1-h[i+k]) for i < N-k
    #             -h[i-k]     for i >= k
    #             0 otherwise
    # We build the gradient of the smooth max w.r.t. h, then map to coeffs.
    grad_h = np.zeros(N)
    for k in range(1, N):
        wk = w[k - 1]
        if wk < 1e-12:
            continue
        # contribution to grad_h from O_k
        # For i in 0..N-1-k: d/dh[i] = (1 - h[i+k])
        grad_h[:N - k] += wk * (1.0 - h[k:])
        # For i in k..N-1: d/dh[i] = -h[i-k]
        grad_h[k:] += wk * (-h[:N - k])

    # Map grad_h to grad_coeffs via the spectral basis
    i = np.arange(N)
    grad_coeffs = np.zeros(2 * n_modes)
    for m in range(1, n_modes + 1):
        c = np.cos(2 * np.pi * m * i / N)
        s = np.sin(2 * np.pi * m * i / N)
        grad_coeffs[2 * m - 2] = 2.0 * np.sum(grad_h * c) / N
        grad_coeffs[2 * m - 1] = 2.0 * np.sum(grad_h * s) / N
    return val, grad_coeffs


def _projected_gradient_step(coeffs, n_modes, t, lr, momentum):
    """One projected gradient step with Nesterov momentum on spectral coeffs."""
    val, grad = _objective_and_grad(coeffs, n_modes, t)
    # Project gradient step onto feasible region via the spectral->h mapping:
    # We take a step in coeff space, then the resulting h is projected to [0,1]
    # and sum-fixed inside _spectral_h_from_coeffs.
    new_coeffs = coeffs - lr * grad
    return new_coeffs, val


def _nesterov_smooth_max(n_modes, t, lr, n_iter, deadline):
    """Nesterov accelerated projected gradient on spectral coefficients."""
    # Initialize with a step-function-like spectrum
    h0 = np.zeros(N)
    h0[N // 2:] = 1.0
    if N % 2 == 1:
        h0[N // 2] = 0.5
    h0 = _fix_sum(h0)
    coeffs = _h_to_spectral_coeffs(h0, n_modes)

    v = np.zeros_like(coeffs)
    x = coeffs.copy()
    best_val = float('inf')
    best_coeffs = coeffs.copy()

    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Nesterov: evaluate at x - v
        x_next = x - v
        val, grad = _objective_and_grad(x_next, n_modes, t)
        if val < best_val:
            best_val = val
            best_coeffs = x_next.copy()
        v = momentum * v + lr * grad
        x = x_next - v
        # Periodically project back to keep magnitudes sane
        if it % 20 == 0:
            h = _spectral_h_from_coeffs(x, n_modes)
            x = _h_to_spectral_coeffs(h, n_modes)

    return best_coeffs, best_val


def _run_slsqp(coeffs, n_modes, t, deadline):
    """Polish with SLSQP on the smooth-max objective."""
    def obj(c):
        val, _ = _objective_and_grad(c, n_modes, t)
        return val

    def grad(c):
        _, g = _objective_and_grad(c, n_modes, t)
        return g

    try:
        result = minimize(
            obj, coeffs, jac=grad, method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        if time.time() < deadline:
            h = _spectral_h_from_coeffs(result.x, n_modes)
            return result.x, _score(h)
    except Exception:
        pass
    return coeffs, _score(_spectral_h_from_coeffs(coeffs, n_modes))


def _optimize_spectral(n_modes, deadline):
    """Full optimization for a given number of spectral modes."""
    # Phase 1: Nesterov projected gradient
    coeffs, val = _nesterov_smooth_max(
        n_modes, t=8.0, lr=0.02, n_iter=300, deadline=deadline
    )
    # Phase 2: SLSQP polish
    coeffs, score = _run_slsqp(coeffs, n_modes, t=12.0, deadline=deadline)
    return coeffs, score


def construct():
    """Construct step heights h in [0,1] with sum=N/2 minimizing max overlap."""
    deadline = time.time() + 140  # safety margin before 150s kill

    # Baseline: step function
    h_step = np.zeros(N)
    h_step[N // 2:] = 1.0
    if N % 2 == 1:
        h_step[N // 2] = 0.5
    h_step = _fix_sum(h_step)
    best_h = h_step.copy()
    best_score = _score(best_h)

    # Try increasing numbers of spectral modes
    for n_modes in [3, 5, 7, 10]:
        if time.time() > deadline - 5:
            break
        try:
            coeffs, score = _optimize_spectral(n_modes, deadline)
            if time.time() > deadline - 2:
                break
            h = _spectral_h_from_coeffs(coeffs, n_modes)
            score = _score(h)
            if score < best_score:
                best_score = score
                best_h = h
        except Exception:
            continue

    # Final coordinate-descent polish on the best h directly
    if time.time() < deadline - 5:
        h = best_h.copy()
        for it in range(200):
            if time.time() > deadline - 2:
                break
            ov = _linear_overlap_values(h)
            k_max = int(np.argmax(ov)) + 1  # shift index
            # Try small perturbations to reduce the worst overlap
            improved = False
            for trial in range(50):
                h2 = h.copy()
                idx = np.random.randint(N)
                delta = np.random.uniform(-0.05, 0.05)
                h2[idx] = np.clip(h2[idx] + delta, 0.0, 1.0)
                h2 = _fix_sum(h2)
                s2 = _score(h2)
                if s2 < best_score - 1e-9:
                    best_score = s2
                    best_h = h2
                    h = h2.copy()
                    improved = True
                    break
            if not improved:
                break

    return best_h
