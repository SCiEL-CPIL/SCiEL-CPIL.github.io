import numpy as np
from scipy.special import genlaguerre, roots_jn, jn_zeros
from scipy.optimize import minimize
import time
import math

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy:
    1. Initialize z_i using the squares of the zeros of J_{1/2}(x), which are (k*pi)^2.
       This aligns with the asymptotic distribution of Laguerre polynomial zeros.
    2. Use a fast floating-point estimator for B (the largest real root of q(x)).
    3. Perform multi-start Nelder-Mead optimization on the z_i values directly,
       with constraints ensuring positivity and distinctness.
    4. Return the optimized z_i.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550.0  # 550s limit, leaving 80s margin for safety
    
    # Pre-compute basis polynomials L_{2k}^{(-0.5)}(x)
    # Store coefficients in lowest-to-highest order for easier evaluation
    basis_polys = []
    basis_derivs = []
    for k in range(26):
        n = 2 * k
        L_n = genlaguerre(n, -0.5)
        coeffs_h2l = L_n.coeffs  # highest to lowest
        # Convert to lowest to highest
        coeffs_l2h = coeffs_h2l[::-1]
        basis_polys.append(coeffs_l2h)
        # Derivative
        deriv_h2l = np.polyder(coeffs_h2l)
        deriv_l2h = deriv_h2l[::-1]
        basis_derivs.append(deriv_l2h)
    
    # Pre-compute constant terms and linear terms for x=0 conditions
    # L_n(0) = binomial(n + alpha, n) = binomial(n - 0.5, n)
    # We can just use the pre-computed polynomials
    L_at_0 = np.array([p[-1] if len(p) > 0 else 0.0 for p in basis_polys])
    Lp_at_0 = np.array([p[-1] if len(p) > 1 else 0.0 for p in basis_derivs])
    
    def estimate_B_fast(z_list):
        """Fast floating-point estimate of B."""
        z_arr = np.array(z_list, dtype=np.float64)
        
        # Build A matrix (26x26)
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        A[0, :] = L_at_0
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        A[1, :] = Lp_at_0
        b[1] = 1.0
        
        # Rows 2..27: g(z_i) = 0, g'(z_i) = 0
        for i in range(12):
            z = z_arr[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            # Evaluate all basis polynomials at z
            for k in range(26):
                A[row_g, k] = np.polyval(basis_polys[k], z)
                A[row_gp, k] = np.polyval(basis_derivs[k], z)
            
            b[row_g] = 0.0
            b[row_gp] = 0.0
        
        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100
        
        # Construct g(x) in lowest-to-highest coefficient form
        g_coeffs = np.zeros(51)
        for k in range(26):
            deg = len(basis_polys[k]) - 1
            # basis_polys[k] is lowest to highest, length deg+1
            g_coeffs[:deg+1] += c[k] * basis_polys[k]
        
        # Construct D(x) = x * prod (x-z_i)^2 in lowest-to-highest form
        # Start with x = [0, 1] (lowest to highest: 0 + 1*x)
        d_coeffs = np.array([0.0, 1.0])
        for z in z_list:
            # (x-z)^2 = z^2 - 2zx + x^2 => [z^2, -2z, 1] in lowest-to-highest
            sq = np.array([z*z, -2*z, 1.0])
            d_coeffs = np.convolve(d_coeffs, sq)
        
        # d_coeffs should have length 25 (degree 24)
        # g_coeffs has length 51 (degree 50)
        # Quotient degree: 50 - 24 = 26
        
        # Convert to highest-to-lowest for np.polydiv
        g_h2l = g_coeffs[::-1]
        d_h2l = d_coeffs[::-1]
        
        try:
            q_h2l, r_h2l = np.polydiv(g_h2l, d_h2l)
        except Exception:
            return 1e100
        
        # Find real roots
        roots = np.roots(q_h2l)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        real_roots = real_roots[real_roots > 0]
        
        if len(real_roots) == 0:
            return 0.0
            
        return float(np.max(real_roots))
    
    def objective(z_params):
        """Objective function: B value for given z_i."""
        # Check validity
        if not np.all(np.isfinite(z_params)):
            return 1e100
        if not np.all(z_params > 0):
            return 1e100
        # Check distinctness
        sorted_z = np.sort(z_params)
        if np.any(np.diff(sorted_z) < 1e-10):
            return 1e100
        
        return estimate_B_fast(z_params)
    
    # Initial guess: Bessel zeros z_i = (k*pi)^2 for k=1..12
    # This corresponds to the zeros of J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    k_vals = np.arange(1, 13, dtype=np.float64)
    z_init = (k_vals * np.pi) ** 2
    
    best_z = z_init.copy()
    best_val = objective(z_init)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 30
    max_iter_per_start = 500
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
            
        if start_idx == 0:
            p0 = z_init.copy()
        else:
            # Perturb the initial guess with random noise
            # Scale of perturbation: relative to the magnitude of z_i
            # z_i ranges from pi^2 ~ 10 to (12*pi)^2 ~ 1400
            # Use log-space perturbation for better scaling
            log_z_init = np.log(z_init)
            log_z_perturbed = log_z_init + np.random.normal(0, 0.1, size=12)
            p0 = np.exp(log_z_perturbed)
            
            # Ensure validity
            if not np.all(p0 > 0) or not np.all(np.isfinite(p0)):
                continue
            sorted_p0 = np.sort(p0)
            if np.any(np.diff(sorted_p0) < 1e-10):
                continue
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': max_iter_per_start, 
                                   'xatol': 1e-8, 
                                   'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x.copy()
        except Exception:
            pass
    
    # Final local refinement with tighter tolerances
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_z, method='Nelder-Mead',
                           options={'maxiter': 1000, 
                                   'xatol': 1e-10, 
                                   'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x.copy()
        except Exception:
            pass
    
    # If optimization failed, use initial guess
    if best_z is None or not np.all(np.isfinite(best_z)):
        best_z = z_init.copy()
    
    # Ensure validity
    if not np.all(best_z > 0) or not np.all(np.isfinite(best_z)):
        best_z = z_init.copy()
    
    # Check distinctness
    sorted_best = np.sort(best_z)
    if np.any(np.diff(sorted_best) < 1e-10):
        best_z = z_init.copy()
    
    return [float(z) for z in best_z]
