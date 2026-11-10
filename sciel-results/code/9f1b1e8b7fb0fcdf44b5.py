# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_score(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    n = 10
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        return 0.0
    P = P / s
    
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y, indexing='xy')  # xx[y,x] = x, yy[y,x] = y
    
    # D = H(x - y)
    val_d = xx - yy
    vals_d = np.unique(val_d)
    p_d = np.array([P[val_d == v].sum() for v in vals_d])
    H_d = _shannon_entropy(p_d)
    
    # M = max(H(x), H(x+y), H(x+2y))
    # H(x)
    vals_x = np.unique(xx)
    p_x = np.array([P[xx == v].sum() for v in vals_x])
    H_x = _shannon_entropy(p_x)
    
    # H(x+y)
    val_xy = xx + yy
    vals_xy = np.unique(val_xy)
    p_xy = np.array([P[val_xy == v].sum() for v in vals_xy])
    H_xy = _shannon_entropy(p_xy)
    
    # H(x+2y)
    val_x2y = xx + 2*yy
    vals_x2y = np.unique(val_x2y)
    p_x2y = np.array([P[val_x2y == v].sum() for v in vals_x2y])
    H_x2y = _shannon_entropy(p_x2y)
    
    M = max(H_x, H_xy, H_x2y)
    if M <= 1e-9:
        return 0.0
    
    return H_d / M

def _neg_score(theta):
    """Negative score for minimization. theta is flattened log-weights."""
    P = np.exp(theta)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        return 0.0
    P = P / s
    return -_compute_score(P)

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second budget
    
    # Start with uniform
    P_best = np.ones((n, n), dtype=np.float64) / (n * n)
    score_best = _compute_score(P_best)
    
    # Parameterization: P[y,x] = exp(theta[y,x]) / Z
    # Start from uniform: theta = 0
    theta0 = np.zeros((n, n), dtype=np.float64)
    
    # Try multiple random initializations and local optimizations
    rng = np.random.default_rng(42)
    
    best_theta = theta0.copy()
    best_score = score_best
    
    # Multi-start local optimization
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate initial theta
        if i == 0:
            theta_init = theta0.copy()
        else:
            # Random perturbation with various scales
            scale = 0.5 + 2.0 * rng.random()
            theta_init = scale * rng.standard_normal((n, n))
        
        # Use L-BFGS-B with bounds to keep theta in reasonable range
        bounds = [(-10, 10)] * (n * n)
        
        try:
            res = minimize(
                _neg_score,
                theta_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            theta_res = res.x.reshape((n, n))
            P_res = np.exp(theta_res)
            P_res = np.clip(P_res, 0, None)
            s = P_res.sum()
            if s > 1e-9:
                P_res = P_res / s
                score_res = _compute_score(P_res)
                if score_res > best_score:
                    best_score = score_res
                    best_theta = theta_res.copy()
        except Exception:
            pass
    
    # Additional refinement: try structured families
    # Family 1: P[y,x] ∝ exp(a*x + b*y)
    # Family 2: P[y,x] ∝ exp(a*x + b*y + c*x*y)
    # Family 3: P[y,x] ∝ exp(a*(x-y)^2 + b*(x+y))
    
    if time.time() < deadline - 20.0:
        # Structured optimization with fewer parameters
        def _neg_score_structured(params):
            a, b, c, d = params
            x = np.arange(n)
            y = np.arange(n)
            xx, yy = np.meshgrid(x, y, indexing='xy')
            theta = a * xx + b * yy + c * xx * yy + d * (xx - yy) ** 2
            P = np.exp(theta)
            P = np.clip(P, 0, None)
            s = P.sum()
            if s <= 1e-9:
                return 0.0
            P = P / s
            return -_compute_score(P)
        
        bounds_struct = [(-5, 5)] * 4
        for trial in range(20):
            if time.time() > deadline - 20.0:
                break
            params0 = 0.5 * rng.standard_normal(4)
            try:
                res = minimize(
                    _neg_score_structured,
                    params0,
                    method='L-BFGS-B',
                    bounds=bounds_struct,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                a, b, c, d = res.x
                x = np.arange(n)
                y = np.arange(n)
                xx, yy = np.meshgrid(x, y, indexing='xy')
                theta = a * xx + b * yy + c * xx * yy + d * (xx - yy) ** 2
                P_struct = np.exp(theta)
                P_struct = np.clip(P_struct, 0, None)
                s = P_struct.sum()
                if s > 1e-9:
                    P_struct = P_struct / s
                    score_struct = _compute_score(P_struct)
                    if score_struct > best_score:
                        best_score = score_struct
                        P_best = P_struct.copy()
            except Exception:
                pass
    
    # Final polish: use the best theta found and do high-precision optimization
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _neg_score,
                best_theta.flatten(),
                method='L-BFGS-B',
                bounds=[(-15, 15)] * (n * n),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            theta_polish = res.x.reshape((n, n))
            P_polish = np.exp(theta_polish)
            P_polish = np.clip(P_polish, 0, None)
            s = P_polish.sum()
            if s > 1e-9:
                P_polish = P_polish / s
                score_polish = _compute_score(P_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    P_best = P_polish.copy()
        except Exception:
            pass
    
    # Ensure P_best is valid
    P_best = np.clip(P_best, 0, None)
    s = P_best.sum()
    if s <= 1e-9:
        P_best = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P_best = P_best / s
    
    return P_best

# EVOLVE-BLOCK-END
