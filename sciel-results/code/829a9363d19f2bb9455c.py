"""
Strategy: DCT Basis Expansion with Softplus Activation and L-BFGS-B Optimization.

Rationale:
1. The current program uses PSO, which is stochastic and lacks gradient information. 
   For a smooth objective function like this one, gradient-based methods are typically 
   more efficient.
2. The non-negativity constraint h >= 0 is handled by representing h as the softplus 
   of a linear combination of DCT basis functions. This ensures h is strictly positive 
   and smooth, which is crucial for L-BFGS-B convergence.
3. DCT basis functions are particularly suitable for 1D signal processing problems 
   like autocorrelation, as they capture smooth variations efficiently.
4. We use L-BFGS-B from scipy.optimize to maximize the objective function. 
   The number of DCT basis functions (K) is chosen to be less than N to reduce 
   dimensionality and encourage smoothness, acting as a form of regularization.
5. We perform multiple restarts with different random initializations of the DCT 
   coefficients to explore the search space and avoid local minima.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Precompute DCT basis matrix
    # We use K basis functions. K < N helps regularize and smooth the solution.
    K = 20 
    x = np.linspace(0, 1, N)
    # DCT-II basis: sqrt(2/N) * cos(pi * k * (n + 0.5) / N)
    # We'll just use cos(pi * k * (n + 0.5) / N) and let the optimizer scale.
    basis = np.zeros((N, K))
    for k in range(K):
        basis[:, k] = np.cos(np.pi * k * (np.arange(N) + 0.5) / N)
    
    # Objective function to minimize (negative of the value)
    # h = softplus(basis @ coeffs)
    def objective(coeffs):
        h = 1.0 / 10.0 * np.log(1.0 + np.exp(10.0 * (basis @ coeffs)))
        # Scale h to a reasonable range? 
        # Softplus outputs are positive. The magnitude of coeffs determines the scale.
        # To help the optimizer, we might want to normalize h or let it find the scale.
        # Let's just evaluate.
        val = evaluate(h)
        if val < 0:
            return 1e6
        return -val

    best_val = -np.inf
    best_h = np.ones(N)
    
    # Restart loop
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize coefficients
        # Small random values
        coeffs0 = np.random.normal(0, 0.1, K)
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, coeffs0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success:
                h_opt = 1.0 / 10.0 * np.log(1.0 + np.exp(10.0 * (basis @ res.x)))
                val = evaluate(h_opt)
                
                if val > best_val:
                    best_val = val
                    best_h = h_opt
                    
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline:
            break

    # If no improvement found, fallback to a good known shape
    # The seed is uniform, which scores 1.0. 
    # A Gaussian or triangular might be better.
    if best_val < 1.0:
        # Try a quick local optimization from a Gaussian start
        # Map Gaussian to coeffs? 
        # Simpler: just run one more L-BFGS-B from a coeffs that approximates a bump.
        # Center of basis functions? 
        # For now, if best_val is low, return a static good shape.
        # Triangular window
        x = np.linspace(0, 1, N)
        h_tri = 1 - np.abs(2*x - 1)
        val_tri = evaluate(h_tri)
        if val_tri > best_val:
            best_h = h_tri
            best_val = val_tri
            
        # Gaussian window
        h_gauss = np.exp(-10*(x-0.5)**2)
        val_gauss = evaluate(h_gauss)
        if val_gauss > best_val:
            best_h = h_gauss
            best_val = val_gauss

    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
