"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Represent the step function as the square of a Chebyshev polynomial series.
f(x) = (sum a_k T_k(x))^2 ensures f(x) >= 0.
Optimize the coefficients a_k using a gradient-based method (L-BFGS-B) to minimize
the normalized max autoconvolution. This enforces non-negativity via the parameterization
(barrier-like effect) and allows for faster convergence than derivative-free methods
on the original space.
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

def get_f_from_coeffs(coeffs, basis):
    """Compute f(x) = (sum c_k T_k(x))^2"""
    p = basis @ coeffs
    return p * p

def objective(coeffs, basis):
    """
    Objective function: normalized max autoconvolution.
    """
    f = get_f_from_coeffs(coeffs, basis)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = np.convolve(f, f, mode='full')
    score = np.max(conv) / (s * s)
    return score

def construct():
    start_time = time.time()
    
    x_grid = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different max orders for the polynomial P(x)
    # f(x) = P(x)^2. If P has order K, f has "order" 2K effectively.
    for max_order in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        n_params = max_order + 1
        basis = chebyshev_basis(x_grid, max_order)
        
        best_params = None
        best_obj = np.inf
        
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            rng = np.random.RandomState(42 + max_order * 100 + restart)
            
            # Initialization
            if restart == 0:
                # Constant function P(x) = 1 => f(x) = 1
                x0 = np.zeros(n_params)
                x0[0] = 1.0
            else:
                # Random coefficients, small magnitude
                x0 = rng.uniform(-1.0, 1.0, n_params)
                x0[0] += 1.0 # Ensure non-zero
            
            # Optimize using L-BFGS-B
            try:
                result = minimize(
                    objective, x0, args=(basis,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                f_test = get_f_from_coeffs(result.x, basis)
                actual_score = compute_score(f_test)
                
                if actual_score < best_obj:
                    best_obj = actual_score
                    best_params = result.x.copy()
                    
            except Exception:
                pass
        
        if best_params is not None:
            f = get_f_from_coeffs(best_params, basis)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score

    # Local refinement on the grid
    if time.time() - start_time < TIME_LIMIT - 20:
        current_h = best_f.copy()
        
        # Normalize to sum = N for numerical stability in local search
        s = np.sum(current_h)
        if s > 1e-12:
            current_h = current_h * (N / s)
        
        # Simple local search: perturb random indices
        for outer_iter in range(200):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            # Pick a few random indices to perturb
            for _ in range(20):
                j = np.random.randint(0, N)
                old_val = current_h[j]
                
                # Try scaling
                for scale in [0.9, 0.8, 1.1, 1.2, 0.7, 1.3]:
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
