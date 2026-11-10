# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize H_{-1} / max(H_0, H_1, H_2) by optimizing a 100-parameter probability distribution.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin (600s limit, 20s buffer)

    # Precompute slope indices
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    # Flatten indices for vectorized operations
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Slope projections
    slopes = [-1.0, 0.0, 1.0, 2.0]
    slope_vals = {}
    for r in slopes:
        slope_vals[r] = X_flat + r * Y_flat
    
    def compute_entropies(P):
        """Compute entropies for all slopes from probability vector P."""
        entropies = {}
        for r in slopes:
            vals = slope_vals[r].astype(int)
            # Use bincount for efficiency
            counts = np.bincount(vals, weights=P)
            # Filter out zero probabilities
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                entropies[r] = 0.0
            else:
                entropies[r] = -np.sum(probs * np.log2(probs))
        return entropies
    
    def objective_logspace(log_params):
        """Objective in log-space for better conditioning."""
        # Numerical stability
        log_params_shifted = log_params - np.max(log_params)
        P = np.exp(log_params_shifted)
        P = P / np.sum(P)
        entropies = compute_entropies(P)
        D = entropies[-1.0]
        M = max(entropies[0.0], entropies[1.0], entropies[2.0])
        if M < 1e-12:
            return 0.0
        return -D / M
    
    # Start from uniform distribution
    P_init = np.ones(n * n) / (n * n)
    log_params_init = np.log(P_init)
    
    best_params = log_params_init.copy()
    best_value = -objective_logspace(log_params_init)  # Store positive value
    
    # Run optimization with multiple restarts
    np.random.seed(42)
    n_restarts = 30
    max_iters_per_restart = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:  # Stop if close to deadline
            break
        
        # Initialize parameters
        if restart == 0:
            log_params = log_params_init.copy()
        else:
            # Perturb from best or random
            if np.random.random() < 0.5:
                log_params = best_params + np.random.randn(n * n) * 0.1
            else:
                log_params = np.random.randn(n * n) * 0.5
        
        # Optimize
        result = minimize(objective_logspace, log_params, method='L-BFGS-B', 
                         options={'maxiter': max_iters_per_restart, 'ftol': 1e-10, 'gtol': 1e-8})
        
        # Check if this is better
        if result.fun < -best_value:
            current_value = -result.fun
            if current_value > best_value:
                best_value = current_value
                best_params = result.x.copy()
        
        if time.time() > deadline - 30:
            break
    
    # Recover best probability distribution
    P_best = np.exp(best_params - np.max(best_params))
    P_best = P_best / np.sum(P_best)
    
    # Reshape to 10x10
    P = P_best.reshape(n, n)
    
    # Ensure validity: clip negative values, normalize
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / (n * n)
    
    # Verify constraints
    assert P.shape == (10, 10)
    assert not np.any(np.isnan(P))
    assert not np.any(np.isinf(P))
    assert not np.any(P < -1e-9)
    assert np.sum(P) > 1e-9
    
    return P

# EVOLVE-BLOCK-END
