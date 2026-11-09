# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values and inverse indices for bincount
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100.
    """
    # Clip to avoid log(0) and ensure positivity for normalization stability
    P_flat = np.clip(P_flat, 1e-15, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    # Project probabilities
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

def _objective(params):
    """
    Objective function for minimization (negative ratio).
    params: 1D array of length 100 (unnormalized probabilities).
    """
    ratio = _compute_ratio(params)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize the 100 cell probabilities directly using differential evolution
    followed by a local polish.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 630s limit
    
    best_ratio = -np.inf
    best_P = np.ones(100) / 100.0
    
    # Initial uniform
    uniform_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(uniform_P)
    best_P = uniform_P.copy()
    
    # Define bounds for differential evolution
    # Each parameter is a probability weight. We can use [0, 1] and normalize later.
    bounds = [(0, 1)] * 100
    
    # Use a smaller population size to fit within time budget, 
    # but enough to explore the 100D space.
    # tol is the tolerance for convergence.
    # maxiter limits the number of generations.
    
    # We will run DE for a significant portion of the time, then polish.
    # DE is expensive in high dimensions. 100 dimensions is quite high.
    # Let's try to be efficient.
    
    # Strategy:
    # 1. Run Differential Evolution with moderate settings.
    # 2. Take the best result and polish with L-BFGS-B or Nelder-Mead.
    
    try:
        # DE settings
        # popsize=15 (default is 15*len(x0) which is 1500, too big). 
        # We need a smaller population for 100 dims to fit in time.
        # Let's try popsize=10 or 15.
        # maxiter=50 or so.
        
        # Callback to check time and update best
        def callback(xk, convergence):
            nonlocal best_ratio, best_P
            if time.time() > deadline - 10:
                # Stop early if we are running out of time
                # Note: DE callback doesn't have a direct 'stop' flag in older scipy, 
                # but we can raise an exception or just let it finish if maxiter is small.
                # A better way is to wrap the objective to check time, but DE manages its own loop.
                # We'll rely on maxiter and popsize to be conservative.
                pass
            
            # Update best if this point is better
            # xk is the current best individual in DE
            r = _compute_ratio(xk)
            if r > best_ratio:
                best_ratio = r
                best_P = xk.copy()
                return False # False means continue, True means stop
            return False

        # Run DE
        # We use a fixed seed for reproducibility in this context, though random is fine.
        res = opt.differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=False, # We will do our own polish to control time
            callback=callback
        )
        
        if res.success or res.fun < 0:
            P_de = res.x
            r_de = _compute_ratio(P_de)
            if r_de > best_ratio:
                best_ratio = r_de
                best_P = P_de.copy()
                
    except Exception as e:
        pass
        
    # Polish the best result found so far
    if time.time() < deadline - 20 and best_ratio > 0:
        try:
            # Use L-BFGS-B for local refinement
            # It requires gradients, but we can use finite differences or just Nelder-Mead.
            # Nelder-Mead is derivative-free and robust for final polish.
            res_polish = opt.minimize(
                _objective, 
                best_P, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_polish.success:
                P_polish = res_polish.x
                r_polish = _compute_ratio(P_polish)
                if r_polish > best_ratio:
                    best_ratio = r_polish
                    best_P = P_polish.copy()
                    
        except Exception:
            pass

    # Final validation and normalization
    P_final = best_P.reshape((10, 10))
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total <= 1e-9:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = P_final / total
        
    return P_final

# EVOLVE-BLOCK-END
