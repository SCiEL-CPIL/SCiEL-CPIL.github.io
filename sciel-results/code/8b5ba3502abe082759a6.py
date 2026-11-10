import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses spectral-domain formulation with precomputed quadrature, multi-restart
    optimization, and high-precision final polish.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs, exact=False) / factorial(10 + m_abs, exact=False))
        return np.sqrt(val)
    
    # m = 0
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, 10] = N * P
    
    # m > 0 and m < 0
    for m in range(1, k + 1):
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, m + 10] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, -m + 10] = sqrt2_N * P * np.sin(m * phi_flat)
    
    # Precompute weighted basis for fast gradient
    B = A * w_flat[:, None]
    
    # 3. Objective and gradient
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = d_phi * np.sum(w_flat * f2)
        norm4_4 = d_phi * np.sum(w_flat * f4)
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        sum_wf_A = d_phi * (B.T @ f_vals)
        sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Sphere-constrained gradient descent polish with adaptive step
    def sphere_polish(c, max_iter=800, lr_init=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = d_phi * np.sum(w_flat * f2)
            norm4_4 = d_phi * np.sum(w_flat * f4)
            if norm2_sq < 1e-12:
                break
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            R_curr = N4 / N2
            
            sum_wf_A = d_phi * (B.T @ f_vals)
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-13:
                break
            
            # Adaptive step size with backtracking
            step = lr
            improved = False
            for _ in range(20):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                f_new = A @ u_new
                f2_new = f_new ** 2
                f4_new = f2_new ** 2
                norm2_sq_new = d_phi * np.sum(w_flat * f2_new)
                norm4_4_new = d_phi * np.sum(w_flat * f4_new)
                if norm2_sq_new < 1e-12:
                    step *= 0.5
                    continue
                R_new = (norm4_4_new ** 0.25) / np.sqrt(norm2_sq_new)
                
                if R_new < R_curr:
                    u = u_new
                    R_curr = R_new
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved or step < 1e-12:
                break
            # Slowly decrease learning rate
            lr = max(lr * 0.999, 0.001)
        
        return u

    # 5. High-precision final polish: use very small steps and many iterations
    def high_precision_polish(c, max_iter=2000, lr_init=0.005):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        lr = lr_init
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = d_phi * np.sum(w_flat * f2)
            norm4_4 = d_phi * np.sum(w_flat * f4)
            if norm2_sq < 1e-12:
                break
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            R_curr = N4 / N2
            
            sum_wf_A = d_phi * (B.T @ f_vals)
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            
            step = lr
            for _ in range(25):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    step *= 0.5
                    continue
                u_new = u_new / u_new_norm
                
                f_new = A @ u_new
                f2_new = f_new ** 2
                f4_new = f2_new ** 2
                norm2_sq_new = d_phi * np.sum(w_flat * f2_new)
                norm4_4_new = d_phi * np.sum(w_flat * f4_new)
                if norm2_sq_new < 1e-12:
                    step *= 0.5
                    continue
                R_new = (norm4_4_new ** 0.25) / np.sqrt(norm2_sq_new)
                
                if R_new < R_curr - 1e-16:
                    u = u_new
                    R_curr = R_new
                    break
                else:
                    step *= 0.5
            
            if step < 1e-15:
                break
            lr = max(lr * 0.9995, 0.0001)
        
        return u

    # 6. Generate diverse initial candidates (reduced count, better structure)
    candidates = []
    rng = np.random.default_rng(123)
    
    # Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Pure m>0 cos modes
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
    
    # Pairs of m modes (all pairs)
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
    
    # Triplets (only low-m combinations)
    for m1 in range(0, 6):
        for m2 in range(m1 + 1, 8):
            for m3 in range(m2 + 1, 10):
                c_triple = np.zeros(n_m)
                c_triple[10 + m1] = 1.0
                c_triple[10 + m2] = 1.0
                c_triple[10 + m3] = 1.0
                c_triple /= np.linalg.norm(c_triple)
                candidates.append(c_triple)
    
    # Random normalized vectors (reduced)
    for _ in range(50):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(30):
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)
    
    # Structured: mix of concentrated and spread patterns
    for alpha in [0.3, 0.5, 0.7]:
        c_struct = np.zeros(n_m)
        for m in range(0, 11):
            c_struct[10 + m] = alpha ** m * rng.standard_normal()
        if np.linalg.norm(c_struct) > 1e-12:
            c_struct /= np.linalg.norm(c_struct)
            candidates.append(c_struct)

    # 7. Euler-Lagrange fixed-point solver
    # At the minimum of R = ||f||_4 / ||f||_2 with ||f||_2 = 1,
    # the Euler-Lagrange condition gives: f(x) = lambda * |f(x)|^2 * sign(f(x))
    # i.e., f is proportional to f^3 pointwise.
    # Fixed-point: compute g = f^3, project onto degree-10 subspace, normalize, over-relax.
    
    def eulag_fixed_point(u, max_iter=5000, omega=0.5, tol=1e-14):
        """Solve the self-consistency equation via over-relaxed fixed-point iteration."""
        u = u.copy()
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            return u
        u = u / u_norm
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            f_vals = A @ u
            # Compute f^3 (the Euler-Lagrange RHS, up to sign)
            f3 = f_vals ** 3
            
            # Project f^3 onto the degree-10 subspace:
            # c_new = A^T * w * f^3 (weighted inner products)
            c_new = d_phi * (B.T @ f3)
            
            # Normalize
            c_new_norm = np.linalg.norm(c_new)
            if c_new_norm < 1e-12:
                break
            u_new = c_new / c_new_norm
            
            # Over-relaxation
            u_relax = omega * u_new + (1 - omega) * u
            u_relax_norm = np.linalg.norm(u_relax)
            if u_relax_norm < 1e-12:
                break
            u_relax = u_relax / u_relax_norm
            
            # Check convergence
            diff = np.linalg.norm(u_relax - u)
            u = u_relax
            if diff < tol:
                break
        
        return u
    
    # 8. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 150:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 9. Optimize top candidates using L-BFGS-B + Euler-Lagrange fixed-point
    best_c = None
    best_val = np.inf
    n_to_optimize = min(10, len(candidate_vals))
    
    for i in range(n_to_optimize):
        if time.time() > deadline - 80:
            break
        val_init, c_init = candidate_vals[i]
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # Apply Euler-Lagrange fixed-point polish
        if best_c is not None and time.time() < deadline - 20:
            fp_result = eulag_fixed_point(best_c, max_iter=3000, omega=0.5, tol=1e-14)
            val_fp, _ = objective_and_grad(fp_result)
            if val_fp < best_val:
                best_val = val_fp
                best_c = fp_result.copy()
            else:
                # Try sphere polish as fallback
                polished = sphere_polish(best_c, max_iter=200, lr_init=0.05)
                val_p, _ = objective_and_grad(polished)
                if val_p < best_val:
                    best_val = val_p
                    best_c = polished.copy()

    # 10. Final high-precision polish: Euler-Lagrange fixed-point + gradient polish
    if best_c is not None and time.time() < deadline - 5:
        # Euler-Lagrange fixed-point with various over-relaxation parameters
        for omega in [0.3, 0.5, 0.7, 1.0]:
            if time.time() > deadline - 5:
                break
            fp_result = eulag_fixed_point(best_c, max_iter=5000, omega=omega, tol=1e-15)
            val_fp, _ = objective_and_grad(fp_result)
            if val_fp < best_val:
                best_val = val_p if False else val_fp
                best_c = fp_result.copy()
        
        # Sphere polish passes
        for lr in [0.01, 0.005, 0.002]:
            if time.time() > deadline - 4:
                break
            polished = sphere_polish(best_c, max_iter=400, lr_init=lr)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()
        
        # Very high precision final pass
        if time.time() < deadline - 3:
            polished = high_precision_polish(best_c, max_iter=1000, lr_init=0.001)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 11. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
