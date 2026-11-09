import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 50

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    l2_sq = np.sum(h * h)
    if l1 == 0.0:
        return 0.0
    # autoconvolution: full linear convolution h*h
    hh = np.convolve(h, h)
    hh_inf = np.max(np.abs(hh))
    if hh_inf == 0.0:
        return 0.0
    return l2_sq / (l1 * hh_inf)

def construct():
    import time
    deadline = time.time() + 120.0

    def safe_score(x):
        try:
            return _score(x)
        except Exception:
            return 0.0

    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score(best_h)

    # Try several structured families and local optimizations
    candidates = []

    # 1) Delta
    candidates.append(best_h.copy())

    # 2) Uniform
    u = np.ones(N)
    candidates.append(u)

    # 3) Triangular (peak in middle)
    t = np.zeros(N)
    mid = N // 2
    for i in range(N):
        t[i] = 1.0 - abs(i - mid) / (mid + 1)
    candidates.append(t)

    # 4) Gaussian-ish
    x = np.linspace(0, 1, N)
    g = np.exp(-10 * (x - 0.5) ** 2)
    candidates.append(g)

    # 5) Two peaks
    two = np.zeros(N)
    two[N // 4] = 1.0
    two[3 * N // 4] = 1.0
    candidates.append(two)

    # 6) Ramp
    r = np.linspace(0.0, 1.0, N)
    candidates.append(r)

    # Evaluate all candidates
    for c in candidates:
        c = np.maximum(c, 0.0)
        if np.sum(c) == 0:
            continue
        s = _score(c)
        if s > best_score:
            best_score = s
            best_h = c.copy()

    # Local optimization via projected gradient ascent on the ratio
    # Normalize h to L1=1 inside the objective to avoid scale invariance issues
    def objective_neg(x):
        # x is raw h; we normalize to L1=1 for scoring
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h = h / s
        return -safe_score(h)

    # Start from best candidate, normalized
    h0 = np.maximum(best_h, 0.0)
    s0 = np.sum(h0)
    if s0 > 0:
        h0 = h0 / s0
    else:
        h0 = np.zeros(N); h0[0] = 1.0

    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    # Constraint: sum(h) == 1
    A = np.ones((1, N))
    cons = LinearConstraint(A, 1.0, 1.0)

    # SLSQP with time guard
    result = None
    try:
        # Use a tighter maxiter and maxfun to respect time
        result = minimize(
            objective_neg, h0,
            method='SLSQP',
            bounds=bounds,
            constraints=[cons],
            options={'maxiter': 500, 'ftol': 1e-10, 'maxfun': 5000},
        )
        if time.time() > deadline - 30:
            result = None
    except Exception:
        result = None

    if result is not None and result.x is not None:
        h_opt = np.maximum(result.x, 0.0)
        s_opt = np.sum(h_opt)
        if s_opt > 0:
            h_opt = h_opt / s_opt
            s_opt_score = _score(h_opt)
            if s_opt_score > best_score:
                best_score = s_opt_score
                best_h = h_opt.copy()

    # Nelder-Mead on the simplex (L1-normalized)
    def nm_objective(x):
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h = h / s
        return -safe_score(h)

    x0 = np.maximum(best_h, 0.0)
    sx = np.sum(x0)
    if sx > 0:
        x0 = x0 / sx
    else:
        x0 = np.zeros(N); x0[0] = 1.0

    try:
        res_nm = minimize(
            nm_objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10},
        )
        if time.time() > deadline - 10:
            res_nm = None
    except Exception:
        res_nm = None

    if res_nm is not None and res_nm.x is not None:
        h_nm = np.maximum(res_nm.x, 0.0)
        snm = np.sum(h_nm)
        if snm > 0:
            h_nm = h_nm / snm
            snm_score = _score(h_nm)
            if snm_score > best_score:
                best_score = snm_score
                best_h = h_nm.copy()

    # Final: ensure nonnegative and return as list
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
