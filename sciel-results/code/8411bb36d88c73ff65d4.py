# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Pre-compute projection indices for fast bincount
N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Projections:
# D is based on x - y
# M is max of entropies for x, x+y, x+2y
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
    # Clip to avoid log(0) and ensure validity
    P_flat = np.clip(P_flat, 1e-15, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    # Project probabilities onto the 4 relevant subspaces
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
    params: 100-dimensional vector of probabilities.
    """
    # params are already constrained to [0, 1] by the optimizer bounds
    # We normalize to ensure sum=1 for the ratio calculation, 
    # though DE usually keeps sum close to 1 if initialized that way.
    # However, to be robust against numerical drift in DE:
    P_flat = np.clip(params, 1e-15, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_norm = P_flat / total
    ratio = _compute_ratio(P_norm)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize the 100 cell probabilities directly using Differential Evolution.
    """
    start_time = time.time()
    # Safety margin: 630s limit. Use 500s for optimization.
    deadline = start_time + 500
    
    # Bounds for each of the 100 probabilities
    # We use [0, 1] for each, and the objective normalizes them.
    # This allows the optimizer to explore relative weights.
    bounds = [(0, 1)] * 100
    
    # Initial best is uniform
    uniform_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(uniform_P)
    best_P = uniform_P.copy()
    
    # Differential Evolution
    # strategy='best1bin' is standard
    # maxiter and popsize control the search breadth
    # polish=True performs L-BFGS-B on the best result found by DE,
    # which is crucial for fine-tuning the continuous landscape.
    
    try:
        # We run DE with a moderate population size.
        # 100 dimensions is high, so DE might be slow, but 500s is generous.
        # popsize=15 (default is 15) -> 1500 evaluations per generation.
        # maxiter=20 -> 30,000 evaluations.
        # Each eval is fast (bincount on 100 items).
        # 30k evals * ~0.0001s = 3s. This is very fast.
        # We can afford much more iterations or larger popsize.
        
        # Let's try to be more thorough:
        # popsize=30, maxiter=50
        # 30 * 50 = 1500 generations? No, popsize is number of individuals.
        # Total evals = popsize * maxiter * (1 + overhead).
        # 30 * 50 = 1500 evals for DE part.
        # Plus polish.
        
        # Actually, DE in scipy is efficient. Let's give it a good shot.
        # We'll use a callback to update best_P and check time.
        
        def callback(xk, convergence):
            nonlocal best_ratio, best_P
            # xk is the current best individual
            # Note: xk is not normalized in the objective context of DE's internal state
            # but our objective normalizes it.
            # Let's compute the ratio for xk directly to track it.
            P_flat = np.clip(xk, 1e-15, None)
            total = np.sum(P_flat)
            if total > 1e-9:
                P_norm = P_flat / total
                r = _compute_ratio(P_norm)
                if r > best_ratio:
                    best_ratio = r
                    best_P = P_norm.copy()
            return False # Do not stop early unless we want to
            
        # Run DE
        res = opt.differential_evolution(
            _objective,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=20,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,  # Use L-BFGS-B for final polishing
            callback=callback,
            init='latinhypercube'
        )
        
        if res.success and res.fun < 0:
            # res.x is the best found by DE+Polish
            P_opt = np.clip(res.x, 1e-15, None)
            total = np.sum(P_opt)
            if total > 1e-9:
                P_opt = P_opt / total
                r_opt = _compute_ratio(P_opt)
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_P = P_opt

    except Exception as e:
        # If DE fails, fallback to the best found so far (likely uniform or early DE result)
        pass

    # If time remains, we could try a second pass or local refinement,
    # but DE+Polish is usually very strong for this type of smooth-ish landscape.
    # The "smooth-ish" part is because entropy is smooth for P > 0.
    # The clipping at 1e-15 introduces non-smoothness at the boundary,
    # but L-BFGS-B in polish should handle the interior well.
    
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
