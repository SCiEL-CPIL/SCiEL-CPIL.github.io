# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

def rudin_problem10() -> np.ndarray:
    """
    Minimizes the L4/L2 norm ratio for degree-10 spherical harmonics using 
    Differential Evolution for global search followed by L-BFGS-B refinement.
    Returns the optimal coefficients c_m for m = -10..10.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    # 1. Setup Grid
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    cos_theta_vals = x_gl
    sin_theta_vals = np.sqrt(np.maximum(0, 1 - cos_theta_vals**2))
    
    # 2. Pre-compute Basis Functions Y_m
    # Y_m values at all grid points. Shape: (n_coeffs, n_grid)
    n_grid = 70 * n_phi
    Y = np.zeros((n_coeffs, n_grid), dtype=np.float64)
    
    # Quadrature weights for the surface integral
    # Integral over sphere: sum_{i,j} f(x_i, phi_j) w_gl_i dphi
    # Weights shape: (n_grid,)
    weights = np.repeat(w_gl, n_phi) * dphi
    
    for m_idx in range(n_coeffs):
        m = m_idx - k
        abs_m = abs(m)
        
        # Normalization constant N_m
        # N_m = sqrt( (21)/(4*pi) * (10-m)! / (10+m)! )
        num = factorial(10 - abs_m)
        den = factorial(10 + abs_m)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * (num / den))
        
        # Associated Legendre Polynomial P_10^|m|(cos_theta)
        P_vals = lpmv(abs_m, 10, cos_theta_vals)
        
        if m == 0:
            Y_vals = np.tile(P_vals[:, np.newaxis], (1, n_phi)) * N_m
        elif m > 0:
            cos_term = np.cos(m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * cos_term[np.newaxis, :])
        else:
            sin_term = np.sin(abs_m * phi)
            Y_vals = (np.sqrt(2.0) * N_m) * (P_vals[:, np.newaxis] * sin_term[np.newaxis, :])
        
        Y[m_idx, :] = Y_vals.flatten()

    # 3. Define Objective and Gradient
    def objective_and_grad(c):
        f = Y.T @ c
        
        f_sq = f**2
        f_four = f_sq**2
        
        A = np.sum(weights * f_sq)  # ||f||_2^2
        B = np.sum(weights * f_four) # ||f||_4^4
        
        if A < 1e-15:
            return 1e10, np.zeros_like(c)
            
        R = (B ** 0.25) / (A ** 0.5)
        
        # Gradient of A
        grad_A = 2.0 * (Y @ (weights * f))
        
        # Gradient of B
        grad_B = 4.0 * (Y @ (weights * f**3))
        
        # grad R
        grad_R = R * (0.25 * grad_B / B - 0.5 * grad_A / A)
        
        return R, grad_R

    def objective_only(c):
        return objective_and_grad(c)[0]

    def grad_only(c):
        return objective_and_grad(c)[1]

    # 4. Global Search using Differential Evolution
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    # Define bounds for DE
    # The ratio is scale-invariant, so we can constrain the L2 norm of coefficients to 1.
    # However, DE works on a box. We will use a box of [-1, 1] and normalize the result
    # before the final refinement, or simply let DE search in [-1, 1] and rely on the 
    # scale invariance to find the direction.
    # Actually, it's safer to search in a box and then project/normalize.
    # Let's search in [-1, 1] for each coefficient.
    bounds = [(-1.0, 1.0)] * n_coeffs
    
    best_c = None
    best_R = np.inf
    
    # Run DE with a reasonable budget
    # maxiter=100, popsize=15 -> ~1500 evaluations.
    # Each evaluation is O(n_coeffs * n_grid) ~ 21 * 9100 ~ 200k ops. Fast.
    # 1500 * 200k = 300M ops. Should take < 1s in numpy.
    
    try:
        if time.time() - start_time < time_limit - safety_margin:
            de_res = differential_evolution(
                objective_only,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                disp=False
            )
            
            if de_res.fun < best_R:
                best_R = de_res.fun
                best_c = de_res.x
    except Exception:
        pass
    
    # If DE failed or didn't run, fall back to random start
    if best_c is None:
        rng = np.random.default_rng(42)
        best_c = rng.normal(0, 1, n_coeffs)
        best_R = objective_only(best_c)

    # 5. Refinement using L-BFGS-B
    # We will run L-BFGS-B from the best DE result.
    # We might also run it from a few other promising DE results if we kept them, 
    # but DE returns only the best. 
    # To be safe, we can also try a few random starts if time permits.
    
    refinement_starts = [best_c]
    
    # Add a couple of random starts for robustness if time permits
    if time.time() - start_time < time_limit - safety_margin - 20:
        rng = np.random.default_rng(123)
        for _ in range(5):
            c_rand = rng.normal(0, 1, n_coeffs)
            refinement_starts.append(c_rand)

    for c_init in refinement_starts:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # Normalize initial guess to unit norm to help convergence
            norm_c = np.linalg.norm(c_init)
            if norm_c > 1e-10:
                c_init = c_init / norm_c
            
            res = minimize(
                objective_only, 
                c_init, 
                jac=grad_only, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x
        except Exception:
            continue

    # Final validity check
    if not np.all(np.isfinite(best_c)):
        return np.ones(n_coeffs, dtype=np.float64)
        
    # Ensure max|c| is within bounds (1e-3, 1e5)
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    return best_c

# EVOLVE-BLOCK-END
