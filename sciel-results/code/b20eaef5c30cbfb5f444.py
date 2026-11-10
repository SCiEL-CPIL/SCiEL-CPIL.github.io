import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy: 
    1. Precompute basis matrix A on quadrature grid.
    2. Define objective R(c) and gradient.
    3. Since R is scale invariant, optimize on the unit sphere ||c||=1.
    4. Use multiple random starts with Nelder-Mead (robust to flat directions) 
       on the normalized coefficients, then polish with BFGS.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin for 630s limit
    
    k = 10
    n_m = 2 * k + 1 # 21 coefficients
    
    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)
    
    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # Precompute weights for norms
    # We want to minimize R = ||f||_4 / ||f||_2
    # Equivalent to minimizing ||f||_4 subject to ||f||_2 = 1
    # ||f||_2^2 = sum w_i f_i^2 d_phi
    # ||f||_4^4 = sum w_i f_i^4 d_phi
    
    # Let's work with the normalized objective:
    # Minimize J(c) = (sum w |Ac|^4 d_phi)^0.25 / (sum w |Ac|^2 d_phi)^0.5
    # To remove scale invariance, we can fix ||c||_2 = 1 and minimize (sum w |Ac|^4 d_phi)^0.25.
    # However, the grid quadrature sum for ||f||_2 is not exactly 1 for ||c||_2=1 due to discretization.
    # It is better to minimize the ratio directly, but the gradient is steeper if we normalize.
    
    def objective_ratio(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        if norm2_sq < 1e-12:
            return 1e10
        return (norm4_4 ** 0.25) / (norm2_sq ** 0.5)

    def objective_normalized(c):
        # Normalize c to unit length
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10
        c_norm = c / norm_c
        f_vals = A @ c_norm
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        # We want to minimize L4/L2. Since c is unit length, 
        # the L2 norm of f is fixed by the basis properties? 
        # No, the basis is orthonormal over the sphere.
        # If the quadrature was exact, sum w f^2 d_phi = 4 pi * ||c||_2^2 = 4 pi.
        # With 70x130, it's very close to 4 pi.
        # So minimizing L4 is sufficient.
        return (norm4_4 ** 0.25)

    # Multi-start strategy
    best_c = None
    best_val = np.inf
    
    # Generate diverse initial guesses
    # 1. Random
    # 2. Single mode (m=0, m=10, etc.)
    # 3. Pairs of modes
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Random starts
    for _ in range(50):
        c_rand = rng.standard_normal(n_m)
        candidates.append(c_rand)
        
    # Single mode starts
    for m in range(-10, 11):
        c_single = np.zeros(n_m)
        c_single[m + 10] = 1.0
        candidates.append(c_single)
        
    # Pair starts (common in flat harmonics)
    for m1 in range(-10, 11, 2):
        for m2 in range(m1 + 1, 11, 2):
            c_pair = np.zeros(n_m)
            c_pair[m1 + 10] = 1.0
            c_pair[m2 + 10] = 1.0
            candidates.append(c_pair)
            
    # Evaluate candidates
    for c in candidates:
        val = objective_ratio(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()
            
    # Local optimization from the best few candidates
    # We will run Nelder-Mead on the normalized objective from the top 10 candidates
    # This is faster and more robust than DE for this manifold
    
    top_candidates = []
    for c in candidates:
        top_candidates.append((objective_ratio(c), c))
    top_candidates.sort(key=lambda x: x[0])
    top_candidates = top_candidates[:10]
    
    for val, c_init in top_candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            # Nelder-Mead on the raw coefficients, but objective normalizes them
            # This effectively searches on the sphere
            res = minimize(
                objective_normalized, 
                c_init, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            c_opt = res.x
            # Check the actual ratio
            val_opt = objective_ratio(c_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # Final Polish with BFGS on the ratio directly (with gradient)
    # This will refine the point found by Nelder-Mead
    
    def objective_and_grad(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients
        w_f = w_flat * f_vals
        w_f3 = w_flat * f_vals ** 3
        
        sum_wf_A = A.T @ w_f * d_phi
        sum_wf3_A = A.T @ w_f3 * d_phi
        
        grad = (N4**3 / N2) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        return R, grad

    if best_c is not None and time.time() < deadline - 10:
        try:
            res_bfgs = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='BFGS', 
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            if res_bfgs.fun < best_val:
                best_c = res_bfgs.x
                best_val = res_bfgs.fun
        except Exception:
            pass

    # Final check on validity
    if best_c is None or not np.all(np.isfinite(best_c)):
        return np.ones(n_m)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
