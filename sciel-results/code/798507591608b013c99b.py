import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _get_harmonics_matrix():
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    
    # phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Weights for quadrature
    W = (2 * np.pi / n_phi) * w[:, np.newaxis]
    W = np.repeat(W, n_phi, axis=1)
    W = W.flatten()
    
    # Build harmonics matrix M of shape (21, n_points)
    n_pts = n_theta * n_phi
    M = np.zeros((2 * k + 1, n_pts), dtype=np.float64)
    
    # Precompute Phi for use in harmonics
    phi_flat = np.tile(phi, n_theta)
    
    for i in range(2 * k + 1):
        m = i - k
        abs_m = abs(m)
        
        N_m = np.sqrt( (2 * k + 1) / (4 * np.pi) * factorial(k - abs_m) / factorial(k + abs_m) )
        
        P_val = lpmv(abs_m, k, x) # shape (70,)
        P_val_flat = np.repeat(P_val, n_phi, axis=0).flatten()
        
        if m == 0:
            Y_val = N_m * P_val_flat
        elif m > 0:
            Y_val = np.sqrt(2) * N_m * P_val_flat * np.cos(m * phi_flat)
        else:
            Y_val = np.sqrt(2) * N_m * P_val_flat * np.sin(abs_m * phi_flat)
            
        M[i, :] = Y_val
        
    return M, W

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    M, W = _get_harmonics_matrix()
    n_coeffs = 21
    n_pts = M.shape[1]
    
    # Precompute weighted M^T for fast gradient
    MW = M * W[None, :]  # shape (21, n_pts)
    
    def compute_R_and_grad(c):
        f = M.T @ c  # shape (n_pts,)
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.dot(W, f2)
        norm4_4 = np.dot(W, f4)
        
        if norm2_sq <= 0 or norm4_4 <= 0:
            return 0.0, np.zeros_like(c)
            
        R = (norm4_4 ** 0.25) / (norm2_sq ** 0.5)
        
        u = MW @ (f ** 3)
        v = MW @ f
        
        grad_R = (1.0 * (norm4_4 ** -0.75) * (norm2_sq ** -0.5) * u) - \
                 (1.0 * (norm4_4 ** 0.25) * (norm2_sq ** -1.5) * v)
        
        return R, grad_R

    def hessian_vec_product(c, v):
        """Compute Hessian-vector product for R."""
        f = M.T @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.dot(W, f2)
        norm4_4 = np.dot(W, f4)
        
        if norm2_sq <= 0 or norm4_4 <= 0:
            return np.zeros_like(c)
            
        # R = A/B where A = (sum W f^4)^{1/4}, B = (sum W f^2)^{1/2}
        # dR/dci = (1/(4 A^3)) * dA/dci * B - A * dB/dci / B^2  ... complex
        
        # Let's use the formula: R = (norm4_4/norm2_sq^2)^{1/4}
        # Let S = norm4_4 / norm2_sq^2, R = S^{1/4}
        # dR/dci = (1/4) S^{-3/4} * dS/dci
        # dS/dci = (d(norm4_4)/dci * norm2_sq^2 - norm4_4 * 2 * norm2_sq * d(norm2_sq)/dci) / norm2_sq^4
        #        = d(norm4_4)/dci / norm2_sq^2 - 2 * norm4_4 * d(norm2_sq)/dci / norm2_sq^3
        
        # d(norm4_4)/dci = 4 * (MW @ f^3)_i = 4 * u_i
        # d(norm2_sq)/dci = 2 * (MW @ f)_i = 2 * v_i
        
        u = MW @ (f ** 3)
        v = MW @ f
        
        # dS/dci = 4*u_i/norm2_sq^2 - 2*norm4_4*2*v_i/norm2_sq^3
        #        = 4*u_i/norm2_sq^2 - 4*norm4_4*v_i/norm2_sq^3
        
        dS_dc = 4.0 * u / (norm2_sq ** 2) - 4.0 * norm4_4 * v / (norm2_sq ** 3)
        
        # dR/dci = (1/4) * S^{-3/4} * dS/dci
        S = norm4_4 / (norm2_sq ** 2)
        if S <= 0:
            return np.zeros_like(c)
        
        dR_dc = 0.25 * (S ** -0.75) * dS_dc
        
        # For Hessian-vector product, we need d(dR_dc)/dc_j * v_j
        # This requires second derivatives, which is complex.
        # For now, approximate with zero (will fall back to gradient)
        return np.zeros_like(c)

    def projected_newton_descent(c_init, max_iter=500, tol=1e-12):
        """Minimize R on the unit sphere using projected Newton method."""
        c = c_init.copy()
        c /= np.linalg.norm(c)
        
        R, grad = compute_R_and_grad(c)
        best_local = c.copy()
        best_R_local = R
        
        step = 0.1
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            grad_tangent = grad - np.dot(grad, c) * c
            
            # Try Newton step if Hessian available, else gradient
            hvp = hessian_vec_product(c, grad_tangent)
            if np.linalg.norm(hvp) > 1e-15:
                # Approximate Newton direction
                d_newton = -hvp
                d_newton = d_newton - np.dot(d_newton, c) * c
                norm_d = np.linalg.norm(d_newton)
                if norm_d > 1e-15:
                    d_newton /= norm_d
                    # Line search for Newton step
                    best_step_n = step
                    best_R_n = R
                    for trial in [step, step * 0.5, step * 0.25]:
                        c_trial = c + trial * d_newton
                        c_trial /= np.linalg.norm(c_trial)
                        R_trial, _ = compute_R_and_grad(c_trial)
                        if R_trial < best_R_n:
                            best_R_n = R_trial
                            best_step_n = trial
                    if best_R_n < R:
                        c = c + best_step_n * d_newton
                        c /= np.linalg.norm(c)
                        R, grad = compute_R_and_grad(c)
                        if R < best_R_local:
                            best_R_local = R
                            best_local = c.copy()
                        continue
            
            # Fallback to gradient step
            best_step = step
            best_R_step = R
            for trial in [step, step * 0.5, step * 0.25, step * 2.0, step * 0.1]:
                c_trial = c - trial * grad_tangent
                c_trial_norm = np.linalg.norm(c_trial)
                if c_trial_norm < 1e-15:
                    continue
                c_trial /= c_trial_norm
                R_trial, _ = compute_R_and_grad(c_trial)
                if R_trial < best_R_step:
                    best_R_step = R_trial
                    best_step = trial
            
            c = c - best_step * grad_tangent
            c /= np.linalg.norm(c)
            R, grad = compute_R_and_grad(c)
            
            if R < best_R_local:
                best_R_local = R
                best_local = c.copy()
            
            if abs(R - best_R_local) < tol and it > 50:
                break
            
            if R > best_R_step * 1.01:
                step *= 0.5
            elif R < best_R_step * 0.99:
                step = min(step * 1.2, 0.5)
        
        return best_local, best_R_local

    # Initial candidates
    c0 = np.zeros(n_coeffs)
    c0[10] = 1.0  # m=0
    R0, _ = compute_R_and_grad(c0)
    best_R = R0
    best_c = c0.copy()
    
    initial_guesses = [c0.copy()]
    
    for m_idx in range(n_coeffs):
        c_single = np.zeros(n_coeffs)
        c_single[m_idx] = 1.0
        initial_guesses.append(c_single)
    
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0 / np.sqrt(2)
            c_pair[j] = 1.0 / np.sqrt(2)
            initial_guesses.append(c_pair)
    
    np.random.seed(42)
    n_random = 30
    for _ in range(n_random):
        c_rand = np.random.randn(n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    for idx, c_init in enumerate(initial_guesses):
        if time.time() > deadline - 30:
            break
        
        c_opt, R_opt = projected_newton_descent(c_init, max_iter=1000, tol=1e-13)
        
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()
    
    if time.time() < deadline - 20:
        constraints = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0}]
        
        def obj(c):
            return compute_R_and_grad(c)[0]
        
        def jac(c):
            return compute_R_and_grad(c)[1]
        
        res = minimize(obj, best_c, jac=jac, method='SLSQP',
                       constraints=constraints,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'maxls': 100})
        
        if res.fun < best_R:
            best_c = res.x
            best_R = res.fun
    
    if best_c is None or np.any(np.isnan(best_c)) or np.any(np.isinf(best_c)):
        best_c = np.zeros(21)
        best_c[10] = 1.0
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = np.zeros(21)
        best_c[10] = 1.0
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1.0

    return best_c
