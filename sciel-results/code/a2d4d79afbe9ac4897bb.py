import numpy as np
import time
from scipy.optimize import minimize

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _params_to_prob(params):
    """
    Convert 18 parameters (log-R, log-C) to a 10x10 probability matrix.
    params[0:10] are log(R), params[10:20] are log(C).
    """
    log_R = params[:10]
    log_C = params[10:20]
    
    # Softmax for stability
    log_R_shifted = log_R - np.max(log_R)
    log_C_shifted = log_C - np.max(log_C)
    
    R = np.exp(log_R_shifted)
    C = np.exp(log_C_shifted)
    
    P = np.outer(R, C)
    
    total = np.sum(P)
    if total < 1e-12:
        return np.ones(100) / 100.0
        
    P_flat = P.flatten() / total
    return P_flat

def _objective(params):
    """
    Objective function for minimization.
    We want to maximize ratio, so we minimize -ratio.
    """
    P = _params_to_prob(params)
    ratio = _compute_ratio(P)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start Nelder-Mead on the 18-dim log-parameter space.
    # The space is continuous and the objective is relatively smooth compared to the 100-dim simplex.
    
    num_starts = 200
    max_iter_per_start = 2000
    max_fev_per_start = 5000
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Generate initial parameters
        # Random log-uniform-ish initialization
        init_params = rng.normal(0, 1, 20)
        
        # Occasionally try a "ridge" initialization in log-space
        # Ridge x + 2y = c implies P ~ exp(-alpha*(x+2y-c)^2)
        # log P = -alpha*(x+2y-c)^2 = -alpha*x^2 - 4alpha*c*x - 2alpha*y^2 - 4alpha*c*y + const
        # This is not exactly rank 1 (cross term x*y missing), but we can approximate.
        # Actually, let's just use random restarts and maybe some structured ones.
        
        # Structured init: Uniform in log space
        if i % 5 == 0:
            init_params = np.zeros(20)
            
        # Structured init: Diagonal concentration (R and C peaked)
        elif i % 5 == 1:
            cx = rng.integers(0, 10)
            cy = rng.integers(0, 10)
            init_params[:10] = -0.5 * np.arange(10)**2
            init_params[10:20] = -0.5 * np.arange(10)**2
            # Shift peaks
            init_params[:10] -= np.max(init_params[:10])
            init_params[10:20] -= np.max(init_params[10:20])
            
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': max_iter_per_start,
                    'maxfev': max_fev_per_start,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            P_opt = _params_to_prob(res.x)
            r_opt = _compute_ratio(P_opt)
            
            if r_opt > best_ratio:
                best_ratio = r_opt
                best_P = P_opt
                
        except Exception:
            pass

    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out
