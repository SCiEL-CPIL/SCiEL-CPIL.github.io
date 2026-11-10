# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    Returns:
        B: Matrix of shape (n_pts, 21) where B[i, j] = Y_j(theta_i, phi_i)
        weights: Array of shape (n_pts,) with quadrature weights
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    # Phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Total points
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    # Precompute P_10^m(x) for m=0..10
    # x has shape (n_theta,)
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    # Normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        # (10-m)! / (10+m)!
        # Use logs or direct calculation for small integers
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    # Build B matrix
    B = np.zeros((n_pts, dim))
    
    # Weights for quadrature
    # Weight for point (i, j) is w[i] * (2*pi/n_phi)
    # We flatten the grid such that index = i * n_phi + j
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    
    # m = 0 -> index 10
    # Y_0 = N[0] * P[0, i]
    # For each i, j, B[i*n_phi+j, 10] = N[0] * P[0, i]
    # P[0, :] has shape (n_theta,). We need to repeat it n_phi times for each i.
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        # Positive m -> index m + 10
        idx_pos = m + 10
        # Y_m = sqrt(2) * N[m] * P[m, i] * cos(m * phi[j])
        cos_term = np.cos(m * phi) # shape (n_phi,)
        val_theta = sqrt2 * N[m] * P[m, :] # shape (n_theta,)
        # Outer product: val_theta[:, None] * cos_term[None, :] -> shape (n_theta, n_phi)
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        
        # Negative m -> index -m + 10
        idx_neg = -m + 10
        # Y_{-m} = sqrt(2) * N[m] * P[m, i] * sin(m * phi[j])
        sin_term = np.sin(m * phi) # shape (n_phi,)
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
        
    return B, weights

# Precompute at module load time (fast enough)
B_GLOBAL, W_GLOBAL = _precompute_basis()

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    c has shape (21,).
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    dim = 21
    k = 10
    
    # Objective function wrapper for scipy
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    # Strategy: Multiple random restarts + local refinement
    # We use Nelder-Mead for robustness on small dimensions
    # We also try some structured initializations
    
    np.random.seed(42)
    
    inits = []
    
    # 1. All ones
    inits.append(np.ones(dim))
    
    # 2. Random normal
    for _ in range(5):
        inits.append(np.random.randn(dim))
        
    # 3. Sparse initializations (single m)
    for i in range(dim):
        c = np.zeros(dim)
        c[i] = 1.0
        inits.append(c)
        
    # 4. Symmetric patterns
    # e.g. only even m, only odd m
    c_even = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 == 0:
            c_even[i] = 1.0
    inits.append(c_even)
    
    c_odd = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 != 0:
            c_odd[i] = 1.0
    inits.append(c_odd)
    
    # 5. Alternating signs
    c_alt = np.array([1.0 if (i - 10) % 2 == 0 else -1.0 for i in range(dim)])
    inits.append(c_alt)
    
    # Evaluate all inits
    for c_init in inits:
        val = objective(c_init)
        if val < best_val:
            best_val = val
            best_coeffs = c_init.copy()
            
    # Local optimization from best inits
    # We will run Nelder-Mead on the top few initializers
    # Sort inits by value
    init_vals = [(objective(c), c) for c in inits]
    init_vals.sort(key=lambda x: x[0])
    
    # Try to optimize from the best 10 starts
    for val, c_start in init_vals[:10]:
        if time.time() > deadline - 30:
            break
            
        # Normalize start to avoid scaling issues (though objective is invariant)
        max_c = np.max(np.abs(c_start))
        if max_c > 0:
            c_norm = c_start / max_c
        else:
            c_norm = c_start
            
        try:
            from scipy.optimize import minimize
            res = minimize(objective, c_norm, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
        except Exception:
            pass

    # Additional random restarts if time permits
    # Use a simpler, faster local search or just more NM restarts
    # Let's do 20 more random restarts
    for i in range(20):
        if time.time() > deadline - 10:
            break
            
        c_rand = np.random.randn(dim)
        # Scale to unit L2 norm approx? No, just random
        # Maybe mix with best solution
        if i % 2 == 0:
            c_rand = 0.5 * best_coeffs + 0.5 * np.random.randn(dim)
            
        try:
            from scipy.optimize import minimize
            res = minimize(objective, c_rand, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final validation and normalization
    coeffs = best_coeffs.astype(np.float64)
    
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
        
    max_c = np.max(np.abs(coeffs))
    
    # Constraints: 1e-3 <= max|c| <= 1e5
    if max_c < 1e-3:
        if max_c > 0:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = np.ones(dim)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
        
    return coeffs

# EVOLVE-BLOCK-END
