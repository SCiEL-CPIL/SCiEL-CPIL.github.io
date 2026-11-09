# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_START = time.time()
_DEADLINE = 600.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _objective(params):
    A, B, C = params
    x = np.arange(_N)
    y = np.arange(_N)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    log_w = A * X + B * Y + C * X * Y
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        return 0.0
    P = w / total
    
    val_d = X - Y
    val_m0 = X
    val_m1 = X + Y
    val_m2 = X + 2 * Y
    
    def _proj_entropy(val, P):
        flat_val = val.flatten()
        flat_P = P.flatten()
        unique_vals, inverse = np.unique(flat_val, return_inverse=True)
        proj_p = np.bincount(inverse, weights=flat_P)
        return _shannon_bits(proj_p)
    
    D = _proj_entropy(val_d, P)
    M0 = _proj_entropy(val_m0, P)
    M1 = _proj_entropy(val_m1, P)
    M2 = _proj_entropy(val_m2, P)
    M = max(M0, M1, M2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def _objective_grad(params):
    """Objective with numerical gradient for L-BFGS-B."""
    A, B, C = params
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0, np.zeros(3)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus)
        params_minus = params.copy()
        params_minus[i] -= eps
        val_minus = _objective(params_minus)
        if not np.isfinite(val_plus) or not np.isfinite(val_minus):
            grad[i] = 0.0
        else:
            grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def arithmetic_kakeya_distribution() -> np.ndarray:
    # Precompute meshgrid
    x = np.arange(_N)
    y = np.arange(_N)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Initial guess: uniform (A=B=C=0)
    x0 = [0.0, 0.0, 0.0]
    
    def _neg_obj(params):
        val = _objective(params)
        if not np.isfinite(val):
            return 0.0
        return -val
    
    def _neg_obj_grad(params):
        val, grad = _objective_grad(params)
        if not np.isfinite(val):
            return 0.0, np.zeros(3)
        return -val, -grad
    
    best_params = x0
    best_val = _objective(best_params)
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Structured starts
    for A in [-1.0, 0.5, 1.0, 2.0]:
        for B in [-1.0, 0.5, 1.0]:
            for C in [-0.5, 0.0, 0.5, 1.0]:
                starts.append([A, B, C])
    
    # Random starts
    for _ in range(20):
        A = rng.uniform(-3, 3)
        B = rng.uniform(-3, 3)
        C = rng.uniform(-2, 2)
        starts.append([A, B, C])
    
    # Optimize from each start using L-BFGS-B (faster than Nelder-Mead for this problem)
    for s in starts:
        if _remaining() < 60:
            break
        try:
            res = minimize(_neg_obj_grad, s, method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerances on the best solution
    if _remaining() > 30:
        try:
            res = minimize(_neg_obj_grad, best_params, method='L-BFGS-B', jac=True,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Generate final P from best params
    A, B, C = best_params
    log_w = A * X + B * Y + C * X * Y
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = w / total
        
    # Final safety checks
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s < 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P

# EVOLVE-BLOCK-END
