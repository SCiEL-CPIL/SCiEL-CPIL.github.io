import numpy as np
import time
import scipy.optimize as opt

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
    probs = probs[probs > 1e-12]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio_smoothed(P_flat, beta=100.0):
    """
    Compute the ratio D / M for a given flat probability vector,
    using a soft-max approximation for M to allow gradient-based optimization.
    """
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
    # Soft-max for M
    entropies = np.array([h_v, h_w, h_z])
    # If beta is very high, this approaches max. If low, approaches mean.
    # We want to start with low beta (smooth) and anneal to high beta (sharp).
    if beta < 1e-5:
        M = np.max(entropies)
    else:
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        # To avoid overflow, subtract max
        max_e = np.max(entropies)
        M = (1.0 / beta) * np.log(np.sum(np.exp(beta * (entropies - max_e)))) + max_e
        
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _compute_ratio_exact(P_flat):
    """
    Compute the exact ratio D / M for evaluation.
    """
    P_flat = np.clip(P_flat, 1e-15, None)
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

def _softmax(z):
    """
    Stable softmax transformation.
    """
    z = np.clip(z, -500, 500)
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z)
    if sum_exp_z == 0:
        return np.ones(100) / 100.0
    return exp_z / sum_exp_z

def _objective(z, beta):
    """
    Objective function for minimization (negative ratio).
    z: 1D array of length 100 (log-probabilities).
    beta: temperature for soft-max.
    """
    P_flat = _softmax(z)
    ratio = _compute_ratio_smoothed(P_flat, beta)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize the 100 cell probabilities using L-BFGS-B with annealed soft-max smoothing.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_ratio = -np.inf
    best_z = np.zeros(100)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.zeros(100))
    
    # 2. Diagonal band
    z_diag = np.full(100, -5.0)
    for i in range(10):
        z_diag[i*10+i] = 5.0
    initial_guesses.append(z_diag)
    
    # 3. Anti-diagonal band
    z_anti = np.full(100, -5.0)
    for x in range(10):
        y = 9 - x
        z_anti[y*10+x] = 5.0
    initial_guesses.append(z_anti)
    
    # 4. Random sparse
    for _ in range(10):
        z_rand = rng.normal(0, 2.0, 100)
        initial_guesses.append(z_rand)
        
    # 5. Structured: high prob on a few specific lines
    # Try lines with different slopes
    for slope in [-1, 0, 1, 2]:
        z_line = np.full(100, -10.0)
        for x in range(10):
            for y in range(10):
                # If point is on line x - slope*y = c? 
                # Let's just pick a random line segment
                pass
        
    # Run optimization for each initial guess
    # We will use an annealing schedule for beta
    # Start with low beta (smooth landscape) and increase to high beta (sharp max)
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        current_z = guess.copy()
        
        # Annealing schedule: beta from 1.0 to 1000.0
        # We do a few L-BFGS-B steps at each beta level
        betas = [1.0, 5.0, 20.0, 100.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                res = opt.minimize(
                    _objective, 
                    current_z, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if res.success or res.fun < 0: # res.fun is -ratio
                    current_z = res.x
            except Exception:
                # If L-BFGS-B fails, try Nelder-Mead as fallback for this beta
                try:
                    res = opt.minimize(
                        _objective, 
                        current_z, 
                        args=(beta,),
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                    current_z = res.x
                except Exception:
                    pass
        
        # Evaluate final result with exact ratio
        P_final_candidate = _softmax(current_z)
        ratio_candidate = _compute_ratio_exact(P_final_candidate)
        
        if ratio_candidate > best_ratio:
            best_ratio = ratio_candidate
            best_z = current_z

    # Final refinement with exact ratio and Nelder-Mead if time permits
    if time.time() < deadline - 30:
        try:
            # Use a tight tolerance Nelder-Mead on the best found z
            # We need to wrap the exact ratio for Nelder-Mead
            def _objective_exact(z):
                P_flat = _softmax(z)
                return -_compute_ratio_exact(P_flat)
            
            res = opt.minimize(
                _objective_exact, 
                best_z, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < -best_ratio:
                best_ratio = -res.fun
                best_z = res.x
        except Exception:
            pass

    # Convert best_z to P
    P_final = _softmax(best_z).reshape((10, 10))
    
    # Final validation and normalization
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total <= 1e-9:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = P_final / total
        
    return P_final

# EVOLVE-BLOCK-END
