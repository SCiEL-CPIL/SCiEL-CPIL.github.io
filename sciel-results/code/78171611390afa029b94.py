# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total


def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = 0.0
    for c in counts_d:
        if c > 0:
            H_d -= c * np.log2(c)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _exp_tilt_P(params):
    """
    Build P from 3-param exponential tilt ansatz: P(x,y) ∝ exp(αx + βy + γxy).
    params = [α, β, γ]
    """
    n = 10
    alpha, beta, gamma = params
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    log_P = alpha * X + beta * Y + gamma * X * Y
    # Shift to avoid underflow
    log_P -= log_P.max()
    P = np.exp(log_P)
    P = _clip_and_normalize(P)
    return P


def _neg_ratio_exp_tilt(params):
    """Negative ratio for minimization of Exp Tilt params."""
    P = _exp_tilt_P(params)
    return -_compute_ratio(P)


def _optimize_exp_tilt(deadline):
    """
    Optimize the 3-parameter exponential tilt ansatz using Differential Evolution.
    Returns best P and ratio found.
    """
    start_time = time.time()
    
    # Bounds for alpha, beta, gamma
    # Values around -2 to 2 are usually sufficient to shift mass to corners
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]
    
    best_P = None
    best_ratio = 0.0
    
    # Use Differential Evolution for robust global search in 3D
    try:
        result = differential_evolution(
            _neg_ratio_exp_tilt,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            tol=1e-8
        )
        
        P_opt = _exp_tilt_P(result.x)
        ratio_opt = _compute_ratio(P_opt)
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
    except Exception:
        pass
    
    # Add some manual restarts if time permits
    if time.time() - start_time < deadline - 60:
        np.random.seed(42)
        for _ in range(5):
            if time.time() - start_time > deadline - 30:
                break
            init = np.random.uniform([-2, -2, -1], [2, 2, 1])
            try:
                res = minimize(_neg_ratio_exp_tilt, init, method='Nelder-Mead', options={'maxiter': 200})
                P_opt = _exp_tilt_P(res.x)
                ratio_opt = _compute_ratio(P_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = P_opt.copy()
            except:
                pass

    if best_P is None:
        return np.ones((10, 10)) / 100.0, 0.0
    
    return best_P, best_ratio


def _compute_ratio_from_d(d):
    """Compute D/M ratio from anti-diagonal weights d."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    
    H_d = 0.0
    for k in range(19):
        if d_norm[k] > 0:
            H_d -= d_norm[k] * np.log2(d_norm[k])
    
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        min_val = 0
        max_val = (n - 1) + r * (n - 1)
        counts = np.zeros(max_val + 1)
        for v in range(max_val + 1):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        h = 0.0
        for c in counts:
            if c > 0:
                h -= c * np.log2(c)
        max_h = max(max_h, h)
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h


def _optimize_anti_diag(d0, deadline):
    """Optimize anti-diagonal weights using Nelder-Mead."""
    start_time = time.time()
    
    # Initial guess
    d = np.ones(19) / 19.0
    best_d = d.copy()
    best_ratio = _compute_ratio_from_d(d)
    
    # Try multiple initializations
    inits = [
        np.ones(19) / 19.0,
        np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1], dtype=float),
        np.array([1,1,1,1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=float),
        np.array([10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10], dtype=float),
    ]
    
    for init in inits:
        if time.time() - start_time > deadline - 60:
            break
        
        d_init = init / init.sum()
        
        try:
            result = minimize(
                lambda d: -_compute_ratio_from_d(d),
                d_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            d_opt = result.x
            d_opt = np.clip(d_opt, 1e-10, None)
            d_opt = d_opt / d_opt.sum()
            
            ratio_opt = _compute_ratio_from_d(d_opt)
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_d = d_opt.copy()
        except Exception:
            pass
    
    return best_d, best_ratio


def _d_to_P(d):
    """Convert anti-diagonal weights to full P matrix."""
    n = 10
    P = np.zeros((n, n))
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    vals_d = X - Y
    
    diag_counts = np.zeros(19)
    for k in range(19):
        diag_counts[k] = np.sum(vals_d == k - 9)
    
    d_norm = d / d.sum()
    
    for k in range(19):
        if d_norm[k] > 0 and diag_counts[k] > 0:
            mask = (vals_d == k - 9)
            P[mask] = d_norm[k] / diag_counts[k]
    
    total = P.sum()
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / 100.0
    
    return P


def _refine_full_powell(P0, deadline):
    """
    Refine P using Powell's method on 100 log-weights.
    P_ij = exp(u_ij) / sum(exp(u))
    """
    start_time = time.time()
    P = P0.copy()
    best_ratio = _compute_ratio(P)
    best_P = P.copy()
    
    # Initialize u from P
    u = np.log(P + 1e-10).flatten()
    
    def _neg_ratio_u(u_flat):
        if time.time() - start_time > deadline - 10:
            return 0.0
        u = u_flat.reshape(10, 10)
        # Softmax
        u_shifted = u - u.max()
        P_temp = np.exp(u_shifted)
        P_temp = P_temp / P_temp.sum()
        return -_compute_ratio(P_temp)
    
    try:
        result = minimize(
            _neg_ratio_u,
            u,
            method='Powell',
            options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-8}
        )
        
        u_opt = result.x.reshape(10, 10)
        u_opt_shifted = u_opt - u_opt.max()
        P_opt = np.exp(u_opt_shifted)
        P_opt = P_opt / P_opt.sum()
        
        ratio_opt = _compute_ratio(P_opt)
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
    except Exception:
        pass
        
    return best_P, best_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses Exponential Tilt ansatz, anti-diagonal weights, and full refinement.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = 580.0  # seconds, with safety margin
    
    best_P = np.ones((n, n)) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Phase 1: Exponential Tilt ansatz optimization (3 params, fast, robust)
    P_tilt, ratio_tilt = _optimize_exp_tilt(deadline)
    if ratio_tilt > best_ratio:
        best_ratio = ratio_tilt
        best_P = P_tilt.copy()
    
    # Phase 2: Optimize anti-diagonal weights (19 params)
    d_best, ratio_diag = _optimize_anti_diag(np.ones(19) / 19.0, deadline)
    P_diag = _d_to_P(d_best)
    if ratio_diag > best_ratio:
        best_ratio = ratio_diag
        best_P = P_diag.copy()
    
    # Phase 3: Full 100-parameter refinement from the best structured solution
    P_refined, ratio_refined = _refine_full_powell(best_P, deadline)
    if ratio_refined > best_ratio:
        best_P = P_refined
        best_ratio = ratio_refined
        
    # Phase 4: Try more random restarts with exponential tilt structure
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > deadline - 60:
            break
        
        # Random Exponential Tilt initialization
        init = np.random.uniform([-2, -2, -1], [2, 2, 1])
        try:
            result = minimize(
                _neg_ratio_exp_tilt,
                init,
                method='Nelder-Mead',
                options={'maxiter': 200}
            )
            P_opt = _exp_tilt_P(result.x)
            ratio_opt = _compute_ratio(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
        except Exception:
            pass
        
        # Quick refinement
        P_ref, ratio_ref = _refine_full_powell(best_P, deadline)
        if ratio_ref > best_ratio:
            best_ratio = ratio_ref
            best_P = P_ref
    
    # Final validation
    best_P = _clip_and_normalize(best_P)
    return best_P


# EVOLVE-BLOCK-END
