import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_N = 10
_START = time.time()
_DEADLINE = 600.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

# Precompute grid and projection indices
_x = np.arange(_N)
_y = np.arange(_N)
_X, _Y = np.meshgrid(_x, _y, indexing='xy')

# Projections
_val_d = _X - _Y
_val_m0 = _X
_val_m1 = _X + _Y
_val_m2 = _X + 2 * _Y

# Map to non-negative integers for bincount
_min_d = _val_d.min()
_inv_d = (_val_d - _min_d).astype(int)
_size_d = int(_val_d.max() - _min_d) + 1

_inv_m0 = _val_m0.astype(int)
_size_m0 = int(_val_m0.max()) + 1

_inv_m1 = _val_m1.astype(int)
_size_m1 = int(_val_m1.max()) + 1

_inv_m2 = _val_m2.astype(int)
_size_m2 = int(_val_m2.max()) + 1

# Flatten indices for fast access
_inv_d_flat = _inv_d.ravel()
_inv_m0_flat = _inv_m0.ravel()
_inv_m1_flat = _inv_m1.ravel()
_inv_m2_flat = _inv_m2.ravel()

def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _compute_entropies(P):
    """
    P is a 10x10 array.
    Returns D, H0, H1, H2.
    """
    p_flat = P.ravel()
    
    p_d = np.bincount(_inv_d_flat, weights=p_flat, minlength=_size_d)
    D = _shannon_bits(p_d)
    
    p_m0 = np.bincount(_inv_m0_flat, weights=p_flat, minlength=_size_m0)
    H0 = _shannon_bits(p_m0)
    
    p_m1 = np.bincount(_inv_m1_flat, weights=p_flat, minlength=_size_m1)
    H1 = _shannon_bits(p_m1)
    
    p_m2 = np.bincount(_inv_m2_flat, weights=p_flat, minlength=_size_m2)
    H2 = _shannon_bits(p_m2)
    
    return D, H0, H1, H2

def _ratio_from_P(P):
    D, H0, H1, H2 = _compute_entropies(P)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def _param_to_P(params):
    """
    params: [A, B, C, D, E, F]
    log_w = A*X + B*Y + C*X*Y + D*X^2 + E*Y^2 + F*X^2*Y^2
    """
    A, B, C, D, E, F = params
    log_w = A * _X + B * _Y + C * _X * _Y + D * _X**2 + E * _Y**2 + F * (_X**2) * (_Y**2)
    # Shift for stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        return np.ones((_N, _N)) / (_N * _N)
    return w / total

def _objective(params):
    P = _param_to_P(params)
    return _ratio_from_P(P)

def _objective_neg(params):
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0
    return -val

def _objective_grad(params):
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0, np.zeros(6)
    
    eps = 1e-7
    grad = np.zeros(6)
    for i in range(6):
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
            
    return -val, -grad

def _coordinate_descent(P, max_iters=100, step_size=0.01):
    """
    Simple coordinate descent on the 100 weights.
    Tries to move mass between pairs of cells to improve the ratio.
    """
    best_val = _ratio_from_P(P)
    P = P.copy()
    
    # Precompute indices
    idxs = np.argwhere(P > 1e-6)
    if len(idxs) == 0:
        return P, best_val

    for _ in range(max_iters):
        if _remaining() < 10:
            break
        improved = False
        # Shuffle order to avoid bias
        np.random.shuffle(idxs)
        
        for i, (r1, c1) in enumerate(idxs):
            # Try moving mass from (r1, c1) to a few neighbors
            # Or just adjust (r1, c1) against a random other cell
            # Let's try adjusting (r1, c1) and (r2, c2) where (r2, c2) is random
            
            # Pick a random target cell
            r2, c2 = np.random.randint(0, _N, 2)
            if r1 == r2 and c1 == c2:
                continue
            
            # Try moving delta from (r2, c2) to (r1, c1)
            # and vice versa
            for delta in [step_size, -step_size]:
                if P[r2, c2] + delta < 0 or P[r1, c1] - delta < 0:
                    continue
                
                P_new = P.copy()
                P_new[r1, c1] -= delta
                P_new[r2, c2] += delta
                
                # Renormalize? No, sum is preserved.
                val = _ratio_from_P(P_new)
                if val > best_val + 1e-9:
                    P = P_new
                    best_val = val
                    improved = True
                    # Update idxs? No, just continue.
                    break
        
        if not improved:
            break
            
    return P, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    # Initial best
    best_params = np.zeros(6)
    best_val = _objective(best_params)
    best_P = _param_to_P(best_params)
    
    # 1. Differential Evolution
    # Bounds for parameters
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    
    try:
        if _remaining() > 60:
            res_de = differential_evolution(_objective_neg, bounds, 
                                            maxiter=50, popsize=20, 
                                            seed=42, tol=1e-8, 
                                            polish=False)
            if -res_de.fun > best_val:
                best_val = -res_de.fun
                best_params = res_de.x
                best_P = _param_to_P(best_params)
    except Exception:
        pass
    
    # 2. Local Optimization (L-BFGS-B) from multiple starts
    # Starts: best from DE, uniform, and some random
    starts = [best_params]
    rng = np.random.default_rng(42)
    for _ in range(20):
        starts.append(rng.uniform(-3, 3, 6))
    
    for s in starts:
        if _remaining() < 20:
            break
        try:
            res = minimize(_objective_neg, s, method='L-BFGS-B', jac=_objective_grad,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
                best_P = _param_to_P(best_params)
        except Exception:
            pass
            
    # 3. Coordinate Descent Polish
    if _remaining() > 30:
        best_P, best_val_cd = _coordinate_descent(best_P, max_iters=200, step_size=0.02)
        if best_val_cd > best_val:
            best_val = best_val_cd
            best_P = best_P
            
        # Try again with smaller step
        if _remaining() > 20:
            best_P, best_val_cd2 = _coordinate_descent(best_P, max_iters=200, step_size=0.005)
            if best_val_cd2 > best_val:
                best_val = best_val_cd2
                best_P = best_P

    # Final safety checks
    best_P = np.clip(best_P, 0.0, None)
    s = np.sum(best_P)
    if s < 1e-9:
        best_P = np.ones((_N, _N)) / (_N * _N)
    else:
        best_P = best_P / s
        
    return best_P
