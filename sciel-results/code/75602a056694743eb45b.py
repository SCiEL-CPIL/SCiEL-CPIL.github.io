"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift).
Structural change: model h as a logistic-map (chaotic) sequence with optimized parameters,
combined with differential evolution for global search and local polish.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
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

# ---- Chaotic sequence generation via logistic map ----

def _logistic_sequence(params, N_pts):
    """Generate a sequence from logistic map.
    params: [r, x0, n_discard, n_use, scale, offset]
    r: control parameter (3.5 to 4.0)
    x0: initial condition (0 to 1)
    n_discard: number of initial iterations to discard (0 to 100)
    n_use: number of points to use (N_pts)
    scale: scaling factor
    offset: offset
    """
    r = params[0]
    x0 = params[1]
    n_discard = int(params[2])
    scale = params[3]
    offset = params[4]
    
    x = x0
    seq = []
    total_needed = n_discard + N_pts
    for i in range(total_needed):
        x = r * x * (1.0 - x)
        if i >= n_discard:
            seq.append(x)
    
    h = np.array(seq[:N_pts]) * scale + offset
    return h

def _logistic_objective(params):
    """Objective for logistic map parameters."""
    h = _logistic_sequence(params, N)
    h = _enforce_constraints(h)
    overlaps = _compute_all_overlaps(h)
    return overlaps.max()

# ---- Differential Evolution global search ----

def _de_search(deadline):
    """Use differential evolution to find good logistic map parameters."""
    bounds = [
        (3.5, 4.0),      # r
        (0.01, 0.99),    # x0
        (0, 100),        # n_discard
        (0.5, 1.5),      # scale
        (-0.2, 0.2),     # offset
    ]
    
    try:
        result = differential_evolution(
            _logistic_objective,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            seed=42,
            polish=True,
            callback=lambda xk, convergence: (time.time() > deadline - 10)
        )
        if result.success or result.fun < _logistic_objective(np.array([3.7, 0.5, 50, 1.0, 0.0])):
            h = _logistic_sequence(result.x, N)
            h = _enforce_constraints(h)
            return h, _overlap_max(h)
    except Exception:
        pass
    
    return None, float('inf')

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

    # 2) Differential Evolution on logistic map parameters (new structural direction)
    if time.time() < deadline - 40:
        h2, s2 = _de_search(deadline)
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
    
    # Add a few chaotic starts
    for r in [3.6, 3.7, 3.8, 3.9, 4.0]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h = _logistic_sequence(np.array([r, x0, 50, 1.0, 0.0]), N)
            h = _enforce_constraints(h)
            starts.append(h)

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
