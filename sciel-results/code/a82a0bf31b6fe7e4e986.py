import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2.
    3. Use Differential Evolution for global search on the unit sphere 
       (by normalizing c inside the objective).
    4. Polish the best result with L-BFGS-B on the normalized objective.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

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
    # Objective: Minimize R(c) = ||f||_4 / ||f||_2
    # f = A @ c
    # ||f||_2^2 = sum w_i f_i^2 d_phi
    # ||f||_4^4 = sum w_i f_i^4 d_phi
    
    def compute_norms(c):
        f_vals = A @ c
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        return norm2_sq, norm4_4, f_vals

    def objective(c):
        # Normalize c to unit length to handle scale invariance
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10
        c_norm = c / norm_c
        norm2_sq, norm4_4, _ = compute_norms(c_norm)
        if norm2_sq < 1e-12:
            return 1e10
        # R = (norm4_4)^0.25 / (norm2_sq)^0.5
        return (norm4_4 ** 0.25) / (norm2_sq ** 0.5)

    # Gradient of the normalized objective for local optimization
    # We optimize g(c) = objective(c / ||c||)
    # Let u = c / ||c||. Then f = A u.
    # R(u) = ||f||_4 / ||f||_2
    # grad_u R = (1/||f||_2) * grad_u ||f||_4 - (||f||_4 / ||f||_2^2) * grad_u ||f||_2
    # grad_u ||f||_4 = 4 ||f||_4^3 * (1/4) * grad_u (||f||_4^4) = ||f||_4^3 * A^T (w f^3) d_phi
    # grad_u ||f||_2 = 1/||f||_2 * grad_u (||f||_2^2) = 1/||f||_2 * 2 A^T (w f) d_phi
    # Actually, let's just use numerical gradient or a simplified analytic one.
    # For L-BFGS-B, numerical gradient is often robust enough if the function is smooth.
    # But let's try to provide an analytic gradient for speed.
    
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Gradients with respect to u
        # d/du (N4) = (1/4) N4^3 * d/du (N4^4) = (1/4) N4^3 * 4 sum(w f^3 A) d_phi = N4^3 sum(w f^3 A) d_phi
        # d/du (N2) = (1/2) N2^-1 * d/du (N2^2) = (1/2) N2^-1 * 2 sum(w f A) d_phi = N2^-1 sum(w f A) d_phi
        
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule: c -> u = c/||c||
        # du/dc = (I - uu^T) / ||c||
        # grad_c R = (I - uu^T) @ grad_u / ||c||
        # (I - uu^T) @ v = v - u (u^T v)
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # Initial candidates for DE
    # DE works on a box. We'll use [-1, 1]^21 and normalize inside.
    bounds = [(-1.0, 1.0) for _ in range(n_m)]
    
    # Run Differential Evolution
    best_c = None
    best_val = np.inf
    
    # Check time
    if time.time() < deadline - 60:
        try:
            # DE can be slow. Use maxiter=100, popsize=15
            # tol=1e-5 is sufficient for global search, we polish later.
            res_de = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-5,
                seed=42,
                init='sobol'  # Better coverage than 'latinhypercube' or 'random'
            )
            if res_de.success or res_de.fun < best_val:
                best_val = res_de.fun
                best_c = res_de.x.copy()
        except Exception as e:
            pass
    
    # If DE failed or timed out, use a few random starts
    if best_c is None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            c_rand = rng.standard_normal(n_m)
            val = objective(c_rand)
            if val < best_val:
                best_val = val
                best_c = c_rand.copy()
    
    # Polish with L-BFGS-B
    if best_c is not None and time.time() < deadline - 10:
        try:
            res_lbfgs = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_c = res_lbfgs.x.copy()
        except Exception as e:
            pass

    # Final validity check
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
