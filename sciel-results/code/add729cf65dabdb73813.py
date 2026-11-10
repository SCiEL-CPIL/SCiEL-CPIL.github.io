import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Use fixed-point iteration with over-relaxation to solve the
       Euler-Lagrange self-consistency equation for the L4/L2 minimizer.
    3. Fall back to L-BFGS-B polish if needed.
    4. Return the best valid coefficient vector found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

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
        
    # 3. Objective and Gradient
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
        
        # Analytic gradient of R with respect to u
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule for c -> u = c/||c||
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Fixed-point iteration for the Euler-Lagrange condition
    # The EL condition: for f with ||f||_2 = 1, we need
    # f(x) = lambda * f(x)^3 / ||f||_4^4
    # i.e., f(x) = (1/lambda) * f(x)^3 / ||f||_4^4
    # In coefficient space: c = P * (A^T @ (w * f^3)) * d_phi / (||f||_4^4)
    # where P is the projection back to the degree-10 subspace
    
    def fixed_point_step(u, over_relax=1.0):
        """One step of fixed-point iteration with over-relaxation."""
        f_vals = A @ u
        f3 = f_vals ** 3
        f4 = f_vals ** 4
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm4_4 < 1e-15:
            return u.copy()
        
        # The new direction is proportional to f^3
        # Project back to coefficient space
        c_new = A.T @ (w_flat * f3) * d_phi / (norm4_4 ** 0.25)
        
        # Normalize to unit L2 norm
        norm2_sq = np.sum(w_flat * (A @ c_new) ** 2) * d_phi
        if norm2_sq < 1e-15:
            return u.copy()
        c_new = c_new / np.sqrt(norm2_sq)
        
        # Over-relaxation
        u_new = (1.0 - over_relax) * u + over_relax * c_new
        
        # Renormalize
        norm_u = np.linalg.norm(u_new)
        if norm_u > 1e-15:
            u_new = u_new / norm_u
        else:
            u_new = u.copy()
            
        return u_new

    # 5. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: Pure m=0 (Axisymmetric)
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Candidates 2-21: Pure m>0 modes (cos and sin parts)
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
        
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
        
    # Candidates 22-41: Pairs of m modes (sum of two different m)
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            if time.time() > deadline - 60:
                break
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
            
    # Candidates 42+: Random normalized vectors with different seeds
    rng = np.random.default_rng(12345)
    n_random = 50
    for _ in range(n_random):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 6. Run fixed-point iteration on each candidate, then polish with L-BFGS-B
    best_c = None
    best_val = np.inf
    
    # First pass: fixed-point iteration on all candidates
    fp_results = []
    for c_init in candidates:
        if time.time() > deadline - 30:
            break
            
        u = c_init.copy()
        u = u / np.linalg.norm(u)
        
        # Run fixed-point with adaptive over-relaxation
        for fp_iter in range(200):
            if time.time() > deadline - 30:
                break
            # Try over-relaxation values
            for over_relax in [0.5, 1.0, 1.5]:
                u_new = fixed_point_step(u, over_relax)
                # Check convergence
                if np.linalg.norm(u_new - u) < 1e-12:
                    u = u_new
                    break
                u = u_new
            
            # Check convergence of the fixed point
            f_vals = A @ u
            f3 = f_vals ** 3
            f4 = f_vals ** 4
            norm4_4 = np.sum(w_flat * f4) * d_phi
            if norm4_4 < 1e-15:
                break
            c_proj = A.T @ (w_flat * f3) * d_phi / (norm4_4 ** 0.25)
            norm_c_proj = np.linalg.norm(c_proj)
            if norm_c_proj > 1e-15:
                c_proj = c_proj / norm_c_proj
                if np.linalg.norm(c_proj - u) < 1e-10:
                    break
        
        # Evaluate
        val, _ = objective_and_grad(u)
        fp_results.append((val, u.copy()))
    
    # Sort by objective value
    fp_results.sort(key=lambda x: x[0])
    
    # Second pass: polish the top fixed-point results with L-BFGS-B
    n_polish = min(20, len(fp_results))
    for val_init, c_init in fp_results[:n_polish]:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()

    # If no good result from fixed-point, fall back to original approach
    if best_c is None or best_val > 1.1:
        for val_init, c_init in fp_results[:5]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    objective_and_grad, 
                    c_init, 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                val = res.fun
                if val < best_val and np.all(np.isfinite(res.x)):
                    best_val = val
                    best_c = res.x.copy()
            except Exception:
                pass

    # 7. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback to m=0
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
