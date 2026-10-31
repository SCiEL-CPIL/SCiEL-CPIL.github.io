"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift).
Combined approach: Fourier-parameterized global search (L-BFGS-B) + full-space
smooth-max SLSQP + coordinate descent polish, with polytope projection.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
X = np.arange(N, dtype=float)

def _overlap_max(h):
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _compute_all_overlaps(h):
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return overlaps

def _project_to_polytope(h, S):
    """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using bisection on lambda."""
    if S < 0 or S > N:
        return np.clip(h, 0, 1)
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > S:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    return np.clip(h - lam, 0.0, 1.0)

def _enforce_constraints(h):
    return _project_to_polytope(h, N / 2.0)

def _smooth_max(values, alpha):
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _smooth_max_weights(values, alpha):
    m = np.max(values)
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    e = np.exp(shifted)
    return e / np.sum(e)

# ---- Full-space optimization (refinement machinery) ----

def _objective(h, alpha):
    return _smooth_max(_compute_all_overlaps(h), alpha)

def _objective_grad(h, alpha):
    overlaps = _compute_all_overlaps(h)
    w = _smooth_max_weights(overlaps, alpha)
    grad = np.zeros(N)
    for i in range(N - 1):
        grad[i] += w[i] * (1.0 - h[i+1])
        grad[i+1] += w[i] * (-h[i])
    grad[N-1] += w[N-1] * 1.0
    return grad

def _optimize_from_start(h0, alpha, max_iter=100):
    n = len(h0)
    target = n / 2.0
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target,
                    'jac': lambda x: np.ones(n)}]
    def obj(x):
        return _objective(x, alpha)
    def obj_grad(x):
        return _objective_grad(x, alpha)
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _coordinate_descent(h, max_iters=300, tol=1e-12, deadline=None):
    h = h.copy()
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            orig = h[i]
            current_max = _overlap_max(h)
            candidates = [orig]
            if 0 < i < N - 1:
                denom = 1.0 + h[i-1] - h[i+1]
                if abs(denom) > 1e-12:
                    candidate = h[i-1] / denom
                    if 0.0 <= candidate <= 1.0:
                        candidates.append(candidate)
                candidates.append(0.0)
                candidates.append(1.0)
            elif i == 0:
                candidates.append(0.0)
            elif i == N - 1:
                denom = 1.0 + h[N-2]
                if denom > 1e-12:
                    candidate = h[N-2] / denom
                    if 0.0 <= candidate <= 1.0:
                        candidates.append(candidate)
                candidates.append(0.0)
                candidates.append(1.0)
            best_candidate = orig
            best_score = current_max
            for cand in candidates:
                h[i] = cand
                score = _overlap_max(h)
                if score < best_score - tol:
                    best_score = score
                    best_candidate = cand
            h[i] = best_candidate
            if best_candidate != orig:
                improved = True
                h = _enforce_constraints(h)
        if not improved:
            break
    return h

# ---- Fourier-parameterized optimization (structural idea) ----

def _build_basis(K):
    """Build cos/sin basis matrix for K modes. Returns (n_params, basis NxP)."""
    P = 2 * K
    basis = np.zeros((N, P))
    for k in range(1, K + 1):
        basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * X / N)
        basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * X / N)
    return basis

def _fourier_search(K, basis, h_init_params, alpha, max_iter, deadline):
    """Optimize Fourier coefficients with L-BFGS-B. Returns best h found."""
    P = 2 * K
    def get_h(p):
        return 0.5 + basis @ p
    def project(h):
        return _project_to_polytope(h, N / 2.0)
    def obj(p):
        h = get_h(p)
        # soft bound penalty to keep h in [0,1]
        over = np.maximum(0.0, h - 1.0)
        under = np.maximum(0.0, -h)
        penalty = 50.0 * (np.sum(over * over) + np.sum(under * under))
        hc = np.clip(h, 0.0, 1.0)
        return _smooth_max(_compute_all_overlaps(hc), alpha) + penalty
    def obj_grad(p):
        h = get_h(p)
        hc = np.clip(h, 0.0, 1.0)
        overlaps = _compute_all_overlaps(hc)
        w = _smooth_max_weights(overlaps, alpha)
        grad_hc = np.zeros(N)
        for i in range(N - 1):
            grad_hc[i] += w[i] * (1.0 - hc[i+1])
            grad_hc[i+1] += w[i] * (-hc[i])
        grad_hc[N-1] += w[N-1] * 1.0
        mask = (h >= 0.0) & (h <= 1.0)
        grad_h = grad_hc * mask
        over = np.maximum(0.0, h - 1.0)
        under = np.maximum(0.0, -h)
        grad_h = grad_h + 100.0 * (over - under)
        return basis.T @ grad_h
    best_h = None
    best_score = float('inf')
    try:
        res = minimize(obj, h_init_params, jac=obj_grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-8})
        h = get_h(res.x)
        h = project(h)
        s = _overlap_max(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    except Exception:
        pass
    # Also record the initial point
    h0 = project(get_h(h_init_params))
    s0 = _overlap_max(h0)
    if s0 < best_score:
        best_score = s0
        best_h = h0.copy()
    return best_h, best_score

def _try_fourier(deadline):
    best_h = None
    best_score = float('inf')
    rng = np.random.default_rng(7)
    for K in [2, 3, 4, 6, 8, 12]:
        if time.time() > deadline - 20:
            break
        basis = _build_basis(K)
        P = 2 * K
        # multiple random starts
        for trial in range(4):
            if time.time() > deadline - 15:
                break
            if trial == 0:
                p0 = np.zeros(P)
            else:
                p0 = rng.uniform(-0.4, 0.4, P)
            for alpha in [100.0, 500.0, 2000.0]:
                if time.time() > deadline - 10:
                    break
                h, s = _fourier_search(K, basis, p0, alpha, max_iter=400, deadline=deadline)
                if h is not None and s < best_score:
                    best_score = s
                    best_h = h.copy()
    return best_h, best_score

# ---- Structured pattern generators (quick warm starts) ----

def _two_point_pattern(p, q):
    h = np.zeros(N)
    h[::2] = p
    h[1::2] = q
    return _enforce_constraints(h)

def _try_two_point_patterns(deadline):
    best_h = None
    best_score = float('inf')
    for p in np.linspace(0, 1, 21):
        for q in np.linspace(0, 1, 21):
            if time.time() > deadline - 30:
                return best_h, best_score
            h = _two_point_pattern(p, q)
            s = _overlap_max(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    return best_h, best_score

def construct():
    deadline = time.time() + 135.0
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)

    # 1) Quick structured warm starts
    h1, s1 = _try_two_point_patterns(deadline)
    if h1 is not None and s1 < best_score:
        best_score = s1
        best_h = h1

    # 2) Fourier-parameterized global search (new structural direction)
    h2, s2 = _try_fourier(deadline)
    if h2 is not None and s2 < best_score:
        best_score = s2
        best_h = h2

    # 3) Build diverse starts for full-space refinement
    rng = np.random.default_rng(42)
    starts = [best_h.copy(), np.full(N, 0.5)]
    alt = np.zeros(N); alt[::2] = 1.0
    starts.append(_enforce_constraints(alt))
    for freq in [1, 2, 3, 5, 8]:
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * X / N)
        starts.append(_enforce_constraints(h))
    for i in range(6):
        starts.append(_enforce_constraints(rng.uniform(0, 1, N)))

    # 4) Full-space smooth-max SLSQP with annealing + coordinate descent polish
    alphas = [20.0, 100.0, 500.0, 2000.0, 10000.0]
    for h0 in starts:
        if time.time() > deadline - 15:
            break
        h = h0.copy()
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            h = _optimize_from_start(h, alpha, max_iter=100)
            h = _enforce_constraints(h)
            s = _overlap_max(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        if time.time() < deadline - 5:
            h = _coordinate_descent(h, max_iters=200, deadline=deadline - 5)
            h = _enforce_constraints(h)
            s = _overlap_max(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        if time.time() < deadline - 5:
            h = _optimize_from_start(h, 20000.0, max_iter=200)
            h = _enforce_constraints(h)
            s = _overlap_max(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # 5) Final global polish on the best solution
    if time.time() < deadline - 2:
        h = _optimize_from_start(best_h, 50000.0, max_iter=300)
        h = _enforce_constraints(h)
        s = _overlap_max(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    if time.time() < deadline - 1:
        h = _coordinate_descent(best_h, max_iters=500, deadline=deadline - 1)
        h = _enforce_constraints(h)
        s = _overlap_max(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    best_h = _enforce_constraints(best_h)
    return best_h
