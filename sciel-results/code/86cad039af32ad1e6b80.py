"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_norms(a0, a4, a8):
    # Coefficients for H0, H4, H8, H12
    # H_k(0) values: 1, -3, 105, -10395
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    
    coeffs = np.array([a0, 0, 0, 0, a4, 0, 0, 0, a8, 0, 0, 0, a12])
    # Indices 0..12
    
    # Norm squared of g: sum a_k^2 * sqrt(pi) * 2^k * k!
    # We can ignore sqrt(pi) as it cancels out in the ratio
    weights = np.array([2.0**k * float(np.math.factorial(k)) for k in range(13)])
    
    norm_g_sq = np.sum(coeffs**2 * weights)
    
    # x * g coefficients b_j
    # x H_k = 0.5 H_{k+1} + k H_{k-1}
    # b_j = 0.5 a_{j-1} + (j+1) a_{j+1}
    b = np.zeros(15) # up to H14
    for k in range(13):
        if coeffs[k] != 0:
            if k+1 < 15:
                b[k+1] += 0.5 * coeffs[k]
            if k-1 >= 0:
                b[k-1] += k * coeffs[k]
    
    norm_xg_sq = np.sum(b**2 * weights[:15])
    
    # g' coefficients d_j
    # H_k' = k H_{k-1}
    # d_j = (j+1) a_{j+1}
    d = np.zeros(13)
    for k in range(13):
        if coeffs[k] != 0 and k > 0:
            d[k-1] += k * coeffs[k]
    
    norm_gp_sq = np.sum(d**2 * weights[:13])
    
    # Objective: sqrt(norm_xg_sq * norm_gp_sq) / norm_g_sq
    if norm_g_sq < 1e-10:
        return 1e10, np.array([a0, a4, a8])
        
    val = np.sqrt(norm_xg_sq * norm_gp_sq) / norm_g_sq
    return val, np.array([a0, a4, a8])

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin from 210s
    
    # Seed values
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_val, best_coeffs = _compute_norms(*seed)
    
    # Multi-start optimization
    n_starts = 50
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial guess
        if i == 0:
            x0 = seed
        else:
            # Perturb around seed or random in a reasonable range
            # Coeffs are roughly: a0 ~ -1, a4 ~ 0.04, a8 ~ 0.0003
            # Let's try random directions
            scale = 1.0 if i < 10 else 0.5
            x0 = seed + rng.normal(0, scale, size=3)
            
        def objective(x):
            val, _ = _compute_norms(x[0], x[1], x[2])
            return val
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success:
                val, coeffs = _compute_norms(*res.x)
                if val < best_val:
                    best_val = val
                    best_coeffs = res.x
        except Exception:
            pass
            
    return best_coeffs
# EVOLVE-BLOCK-END
