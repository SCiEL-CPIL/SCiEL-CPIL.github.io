import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses spectral-domain formulation with precomputed quadrature, multi-restart
    optimization, and Basin Hopping for global search escape.
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

    # 4. Fast local minimizer on the sphere (Gradient Descent with Backtracking)
    def local_minimize_sphere(c, max_iter=200, lr_init=0.05):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c, 1e10
        u = c / norm_c
        lr = lr_init
        R_curr = 1e10
        # Initial eval
        val, _ = objective_and_grad(u)
        R_curr = val
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
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
            lr = max(lr * 0.999, 0.001)
        
        return u, R_curr

    # 5. Basin Hopping on the Sphere
    def basin_hopping_sphere(c_init, max_iter=50, kick_scale=0.5, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        
        u_curr, R_curr = local_minimize_sphere(c_init, max_iter=100, lr_init=0.05)
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Kick
            noise = rng.standard_normal(n_m)
            u_kicked = u_curr + kick_scale * noise
            u_kicked = u_kicked / np.linalg.norm(u_kicked)
            
            # Minimize kicked point
            u_new, R_new = local_minimize_sphere(u_kicked, max_iter=100, lr_init=0.05)
            
            # Accept if better or with Metropolis probability (here we just accept if better to be safe/fast, 
            # or use T=0 for strict descent which is safer for time limits)
            if R_new < R_curr:
                u_curr = u_new
                R_curr = R_new
            else:
                # If not better, we might still accept with prob exp(-dR/T) but for simplicity and speed, 
                # we can just retry with smaller kick or stop if stuck.
                # Let's try a smaller kick next time if we fail? 
                # For now, strict descent is robust.
                pass
            
        return u_curr, R_curr

    # 6. Generate diverse initial candidates
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
    
    # All triplets
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                c_triple = np.zeros(n_m)
                c_triple[10 + m1] = 1.0
                c_triple[10 + m2] = 1.0
                c_triple[10 + m3] = 1.0
                c_triple /= np.linalg.norm(c_triple)
                candidates.append(c_triple)
    
    # Random normalized vectors
    for _ in range(100):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(60):
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)

    # 7. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 120:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Optimize top candidates
    best_c = None
    best_val = np.inf
    n_to_optimize = min(30, len(candidate_vals))
    
    for i in range(n_to_optimize):
        if time.time() > deadline - 40:
            break
        val_init, c_init = candidate_vals[i]
        
        # 1. L-BFGS-B
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # 2. Sphere Polish (Gradient Descent)
        if best_c is not None and time.time() < deadline - 20:
            polished, val_p = local_minimize_sphere(best_c, max_iter=500, lr_init=0.05)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 9. Basin Hopping on the best candidate found so far
    if best_c is not None and time.time() < deadline - 30:
        # Run BH with a few different kick scales
        for kick_scale in [0.5, 0.2, 0.1]:
            if time.time() > deadline - 15:
                break
            bh_c, bh_val = basin_hopping_sphere(best_c, max_iter=30, kick_scale=kick_scale, rng=np.random.default_rng(42+kick_scale*100))
            if bh_val < best_val:
                best_val = bh_val
                best_c = bh_c.copy()
        
        # Final polish
        if time.time() < deadline - 5:
            polished, val_p = local_minimize_sphere(best_c, max_iter=1000, lr_init=0.005)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 10. Final Validity Check and Scaling
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
