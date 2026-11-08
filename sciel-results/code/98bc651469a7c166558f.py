"""
Strategy: Dimensionality reduction via parametric basis functions.
Instead of optimizing 50 independent heights, we represent h as a linear
combination of a small number of smooth basis functions (e.g., shifted
Gaussians or B-splines) plus a constant offset. This reduces the search
space from 50 dimensions to ~10-15, making it easier for gradient-based
methods (L-BFGS-B) to find global optima when combined with multi-start.
We use a smooth surrogate for the objective to handle the non-differentiable
norm_inf term, then polish with Nelder-Mead on the exact function.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

# Precompute basis functions
# We use a set of shifted, scaled Gaussians centered at different positions
# This allows capturing peaks and valleys with few parameters.
# We also include a constant basis for overall scale.
X_GRID = np.linspace(0, N - 1, N)
NUM_BASIS = 8
BASIS = np.zeros((NUM_BASIS, N))
BASIS[0] = 1.0  # Constant
for i in range(1, NUM_BASIS):
    # Centers spread across the domain
    center = (i / NUM_BASIS) * (N - 1)
    # Width varies slightly to allow sharp or broad peaks
    width = N / (NUM_BASIS * 1.5)
    BASIS[i] = np.exp(-0.5 * ((X_GRID - center) / width) ** 2)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _param_to_h(params):
    """Convert parameter vector to heights h, ensuring non-negativity."""
    # params[0] is scale, params[1:] are basis coefficients
    scale = params[0]
    coeffs = params[1:]
    h = scale * (BASIS.T @ coeffs)
    # Enforce non-negativity by rectifying
    h = np.maximum(0.0, h)
    return h

def _smooth_value(params, beta=20.0):
    """
    Smooth surrogate for the value function.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    Returns negative value for minimization.
    """
    h = _param_to_h(params)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Smooth max using log-sum-exp
    # We want to approximate max(|g|). 
    # Since g can be negative, we use log-sum-exp on |g|.
    abs_g = np.abs(g)
    # Avoid log(0)
    max_g = np.max(abs_g)
    if max_g == 0:
        return 0.0
    # log-sum-exp trick
    shifted = abs_g / max_g
    lse = np.log(np.sum(np.exp(beta * shifted))) / beta + np.log(max_g) / beta
    norm_inf_smooth = np.exp(lse)
    
    if norm_1 <= 0 or norm_inf_smooth <= 0:
        return 0.0
        
    val = l2 / (norm_1 * norm_inf_smooth)
    return -val

def _neg_value_exact(params):
    """Negative of the exact value for minimization algorithms."""
    h = _param_to_h(params)
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0) # 10s safety margin

    best_val = 0.0
    best_params = np.zeros(NUM_BASIS)
    best_params[0] = 1.0 # Start with uniform height 1
    
    # Initial evaluation
    best_val = compute_value(_param_to_h(best_params))
    
    # Multi-start L-BFGS-B on smooth surrogate
    # We generate several diverse starting points in the parameter space
    num_starts = 20
    bounds = [(0.1, 10.0)] + [(-5.0, 5.0)] * (NUM_BASIS - 1)
    
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random start
        params = np.zeros(NUM_BASIS)
        params[0] = RNG.uniform(0.5, 5.0)
        params[1:] = RNG.uniform(-2.0, 2.0, size=NUM_BASIS - 1)
        
        # Try to optimize
        try:
            res = opt.minimize(
                _smooth_value,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Evaluate exact value at the result
            if res.fun < 0:
                v = -res.fun
                # Also check the exact value to be safe
                h_res = _param_to_h(res.x)
                v_exact = compute_value(h_res)
                if v_exact > best_val:
                    best_val = v_exact
                    best_params = res.x.copy()
        except Exception:
            pass

    # Polish Phase: Nelder-Mead on the exact function using the best params
    if time.time() < deadline - 5.0:
        try:
            # We need to convert params back to h for NM, or just optimize params with exact function
            # Optimizing params is better as it stays in the low-dim manifold
            res_nm = opt.minimize(
                _neg_value_exact,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res_nm.fun < 0:
                v = -res_nm.fun
                if v > best_val:
                    best_val = v
                    best_params = res_nm.x.copy()
        except Exception:
            pass

    # Final conversion to h
    best_h = _param_to_h(best_params)
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    # One last check to ensure we return the absolute best we've seen
    final_val = compute_value(best_h)
    if final_val > best_val:
        best_val = final_val

    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
