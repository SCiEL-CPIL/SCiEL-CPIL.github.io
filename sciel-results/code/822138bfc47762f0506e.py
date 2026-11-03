"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as a linear combination of Chebyshev polynomials.
Optimize the coefficients using a derivative-free algorithm (Powell) with a penalty
term for non-negativity violations. This avoids the issues with gradients of the
non-smooth max objective and hard clipping.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def chebyshev_basis(x, max_order):
    """
    Generate Chebyshev polynomial basis values for orders 0 to max_order.
    x should be in [-1/4, 1/4]. Returns array of shape (len(x), max_order+1)
    """
    x_scaled = x / 0.25  # Map to [-1, 1]
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    
    n_terms = max_order + 1
    basis = np.zeros((len(x), n_terms))
    
    # T_0(x) = 1
    basis[:, 0] = 1.0
    # T_1(x) = x
    if n_terms > 1:
        basis[:, 1] = x_scaled
        
    for k in range(2, n_terms):
        basis[:, k] = 2 * x_scaled * basis[:, k-1] - basis[:, k-2]
        
    return basis

def objective(coeffs, max_order, x_grid, basis, penalty_weight):
    """
    Objective function: normalized max autoconvolution + penalty for negative values.
    """
    f = basis @ coeffs
    min_val = np.min(f)
    
    # Penalty for non-negativity
    penalty = 0.0
    if min_val < 0:
        # Quadratic penalty on the magnitude of negative parts
        neg_parts = f[f < 0]
        penalty = penalty_weight * np.sum(neg_parts**2)
    
    # Compute score
    # We use the raw f for convolution, but if there are significant negatives,
    # the score might be misleading. The penalty drives them to 0.
    # For the score calculation, we clip to 0 to match the task definition of "nonnegative step function"
    f_clipped = np.maximum(f, 0.0)
    s = np.sum(f_clipped)
    if s < 1e-15:
        return 1e10 + penalty
    
    conv = np.convolve(f_clipped, f_clipped, mode='full')
    score = np.max(conv) / (s * s)
    
    return score + penalty

def construct():
    start_time = time.time()
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders
    # Even orders are often sufficient for symmetric functions, but let's allow all orders
    # to capture any asymmetry if it helps (though the problem is symmetric, optimal is likely symmetric).
    # We will enforce symmetry by only using even orders to reduce parameters and respect symmetry.
    # If we use only even orders, f(x) = f(-x).
    
    for max_order in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        # Use only even orders: 0, 2, 4, ..., max_order
        # This ensures f is even.
        even_orders = list(range(0, max_order + 1, 2))
        n_params = len(even_orders)
        
        # Build basis for these specific orders
        full_basis = chebyshev_basis(x_grid, max_order)
        basis = full_basis[:, even_orders]
        
        best_params = None
        best_obj = np.inf
        
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            rng = np.random.RandomState(42 + max_order * 100 + restart)
            
            # Initialization
            # Start with a constant function (c0=1, others 0) or random
            if restart == 0:
                x0 = np.zeros(n_params)
                x0[0] = 1.0
            else:
                # Random small coefficients
                x0 = rng.uniform(-0.5, 0.5, n_params)
                x0[0] += 1.0 # Ensure constant term is large enough to start positive
            
            # Optimize using Powell (derivative-free, robust to non-smoothness)
            try:
                result = minimize(
                    objective, x0, args=(max_order, x_grid, basis, 1000.0),
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-8, 'xtol': 1e-8}
                )
                
                # Check if the solution is valid (non-negative)
                f_test = basis @ result.x
                if np.min(f_test) < -1e-8:
                    # If it's slightly negative, clip and re-evaluate
                    f_test = np.maximum(f_test, 0.0)
                
                actual_score = compute_score(f_test)
                
                if actual_score < best_obj:
                    best_obj = actual_score
                    best_params = result.x.copy()
                    
            except Exception:
                pass
        
        if best_params is not None:
            f = basis @ best_params
            f = np.maximum(f, 0.0)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement on the grid
    if time.time() - start_time < TIME_LIMIT - 20:
        current_h = best_f.copy()
        
        # Normalize
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        # Simple local search: perturb random indices
        for outer_iter in range(100):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            # Pick a few random indices to perturb
            for _ in range(10):
                j = np.random.randint(0, N)
                old_val = current_h[j]
                
                # Try scaling
                for scale in [0.9, 0.8, 1.1, 1.2]:
                    h_new = current_h.copy()
                    h_new[j] = old_val * scale
                    h_new = np.maximum(h_new, 0.0)
                    
                    s_new = np.sum(h_new)
                    if s_new < 1e-12:
                        continue
                    h_new = h_new * (N / s_new)
                    
                    conv_new = np.convolve(h_new, h_new, mode='full')
                    val_new = np.max(conv_new) / (N * N)
                    
                    if val_new < best_score:
                        current_h = h_new
                        best_score = val_new
                        best_f = current_h.copy()
                        improved = True
                        break
            if not improved:
                break
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
