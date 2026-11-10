# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time


def _fixed_point_iteration(A, B, w_flat, d_phi, c_init, max_iter=2000, tol=1e-10, time_budget=300.0):
    """
    Fixed-point iteration for the Euler-Lagrange condition of minimizing ||f||_4/||f||_2.
    
    The EL condition states that for the optimal f (L2-normalized):
        f = alpha * f^3 / ||f^3||_2  (pointwise, up to normalization)
    
    We iterate: u_new = normalize( A^T * (f^3 * weights) )
    with over-relaxation for stability.
    """
    start_time = time.time()
    u = c_init.copy()
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        return u, np.inf
    u = u / u_norm
    
    best_u = u.copy()
    best_R = np.inf
    omega = 0.3  # Over-relaxation parameter
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
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
        R = N4 / N2
        
        if R < best_R:
            best_R = R
            best_u = u.copy()
        
        # Compute the "target" direction: A^T (w * f^3)
        f3 = f_vals ** 3
        target = d_phi * (B.T @ f3)
        
        # Normalize target
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-12:
            break
        target = target / target_norm
        
        # Check convergence
        diff = np.linalg.norm(target - u)
        if diff < tol:
            break
        
        # Over-relaxed update
        u_new = u + omega * (target - u)
        u_new_norm = np.linalg.norm(u_new)
        if u_new_norm < 1e-12:
            break
        u = u_new / u_new_norm
        
        # Adaptive over-relaxation
        if it % 50 == 0 and it > 0:
            # Try to increase omega if progress is good
            if R < best_R * (1 - 1e-6):
                omega = min(omega * 1.1, 0.8)
            else:
                omega = max(omega * 0.8, 0.1)
    
    return best_u, best_R


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a degree-10 spherical harmonic
    that minimize R = ||f||_4 / ||f||_2.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

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

    # 3. Objective and Gradient (sphere-constrained formulation)
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
        
        # Gradient on the sphere
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Sphere-constrained gradient descent polish
    def sphere_polish(c, max_iter=500, lr=0.05):
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
            if grad_norm < 1e-12:
                break
            
            step = lr
            R_curr = N4 / N2
            for _ in range(15):
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
                    break
                else:
                    step *= 0.5
            
            if step < 1e-10:
                break
        
        return u

    # 5. Generate diverse initial candidates
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
    
    # Pairs of m modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
    
    # Triplets
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            for m3 in range(m2 + 1, 11):
                if time.time() > deadline - 180:
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
        if time.time() > deadline - 180:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Random with emphasis on low-m modes
    for _ in range(60):
        if time.time() > deadline - 180:
            break
        c_rand = np.zeros(n_m)
        for m in range(0, 11):
            if rng.random() < 0.7:
                c_rand[10 + m] = rng.standard_normal()
        if np.linalg.norm(c_rand) > 1e-12:
            c_rand /= np.linalg.norm(c_rand)
            candidates.append(c_rand)

    # 6. Evaluate and sort candidates
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 90:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    # 7. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    for val_init, c_init in candidate_vals:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()
        
        # Fixed-point iteration for EL condition
        if time.time() < deadline - 30:
            fp_start = best_c if best_c is not None else c_init
            fp_u, fp_R = _fixed_point_iteration(
                A, B, w_flat, d_phi, fp_start, 
                max_iter=1500, tol=1e-12, 
                time_budget=min(60.0, deadline - time.time() - 15)
            )
            if fp_R < best_val:
                best_val = fp_R
                best_c = fp_u.copy()
        
        # Sphere polish on the best result so far
        if best_c is not None and time.time() < deadline - 10:
            polished = sphere_polish(best_c, max_iter=400, lr=0.05)
            val_p, _ = objective_and_grad(polished)
            if val_p < best_val:
                best_val = val_p
                best_c = polished.copy()

    # 8. Final polish on the best candidate
    if best_c is not None and time.time() < deadline - 5:
        polished = sphere_polish(best_c, max_iter=600, lr=0.02)
        val_p, _ = objective_and_grad(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()
        
        # Final fixed-point iteration for EL condition
        if time.time() < deadline - 3:
            fp_u, fp_R = _fixed_point_iteration(
                A, B, w_flat, d_phi, best_c,
                max_iter=2000, tol=1e-14,
                time_budget=max(10.0, deadline - time.time() - 2)
            )
            if fp_R < best_val:
                best_val = fp_R
                best_c = fp_u.copy()

    # 9. Final validation
    if best_c is None:
        best_c = np.ones(n_m, dtype=np.float64)
    
    if np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.ones(n_m, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = np.ones(n_m, dtype=np.float64)
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1e5
    
    return best_c


# EVOLVE-BLOCK-END
