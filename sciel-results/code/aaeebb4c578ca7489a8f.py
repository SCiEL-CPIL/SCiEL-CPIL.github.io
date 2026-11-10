# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses sphere-constrained gradient descent with exact analytic gradients
    and diverse initial candidates exploiting m-mode symmetry.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

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

    # 3. Objective and Gradient (sphere-constrained: normalize c internally)
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

    # 4. Sphere-constrained gradient descent polish with line search
    def sphere_polish(c, max_iter=1500, lr_init=0.08):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        
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
            
            sum_wf_A = d_phi * (B.T @ f_vals)
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-15:
                break
            
            step = lr_init
            R_curr = N4 / N2
            improved = False
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
                
                if R_new < R_curr:
                    u = u_new
                    R_curr = R_new
                    improved = True
                    break
                else:
                    step *= 0.5
            
            if not improved:
                step *= 0.5
                if step < 1e-14:
                    break
            # Slowly decay learning rate
            lr_init = max(0.005, lr_init * 0.998)
        
        return u

    # 5. High-precision final polish with very tight tolerances
    def high_precision_polish(c, max_iter=2000):
        c = c.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        
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
            
            sum_wf_A = d_phi * (B.T @ f_vals)
            sum_wf3_A = d_phi * (B.T @ (f_vals ** 3))
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-16:
                break
            
            step = 0.02
            R_curr = N4 / N2
            improved = False
            for _ in range(30):
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
            
            if not improved:
                step *= 0.5
                if step < 1e-16:
                    break
        
        return u

    # 6. Generate diverse initial candidates
    candidates = []
    
    # Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Pure m>0 cos modes
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
    
    # Pure m<0 sin modes
    for m in range(1, 11):
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
    
    # Pairs of m modes (cos+cos, cos+sin, sin+sin)
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            # cos + cos
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
            # cos + sin
            c_pair2 = np.zeros(n_m)
            c_pair2[10 + m1] = 1.0
            c_pair2[-m2 + 10] = 1.0
            c_pair2 /= np.linalg.norm(c_pair2)
            candidates.append(c_pair2)
    
    # Triplets of m modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                if time.time() > deadline - 250:
                    break
                c_triple = np.zeros(n_m)
                c_triple[10 + m1] = 1.0
                c_triple[10 + m2] = 1.0
                c_triple[10 + m3] = 1.0
                c_triple /= np.linalg.norm(c_triple)
                candidates.append(c_triple)
    
    # Random normalized vectors
    rng = np.random.default_rng(123)
    for _ in range(100):
        if time.time() > deadline - 250:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(60):
        if time.time() > deadline - 250:
            break
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.6:
                c_rand[10 + m] = rng.standard_normal()
            if m > 0 and rng.random() < 0.4:
                c_rand[-m + 10] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)

    # 7. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 150:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 8. Optimize Each Candidate with L-BFGS-B then sphere polish
    best_c = None
    best_val = np.inf
    
    for val_init, c_init in candidate_vals:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            val = res.fun
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass
    
    # 9. Final sphere-constrained polish
    if best_c is not None and time.time() < deadline - 10:
        polished = sphere_polish(best_c, max_iter=1500)
        val_polished, _ = objective_and_grad(polished)
        if np.isfinite(val_polished) and val_polished < best_val:
            best_val = val_polished
            best_c = polished
    
    # 10. High-precision final polish
    if best_c is not None and time.time() < deadline - 5:
        hp_polished = high_precision_polish(best_c, max_iter=2000)
        val_hp, _ = objective_and_grad(hp_polished)
        if np.isfinite(val_hp) and val_hp < best_val:
            best_val = val_hp
            best_c = hp_polished

    # 11. Validate and scale
    if best_c is None:
        best_c = np.ones(n_m, dtype=np.float64)
    else:
        best_c = np.asarray(best_c, dtype=np.float64)
    
    if best_c.shape != (n_m,):
        best_c = np.ones(n_m, dtype=np.float64)
    
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_m, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1e-2 if max_abs > 0 else np.ones(n_m, dtype=np.float64)
    if max_abs > 1e5:
        best_c = best_c / max_abs * 1e4
    
    return best_c


# EVOLVE-BLOCK-END
