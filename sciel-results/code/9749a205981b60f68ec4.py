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

    def project_to_sphere(c):
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.zeros_like(c)
        return c / norm

    def projected_gradient_descent(c_init, max_iter=2000, tol=1e-14):
        """Minimize R on the unit sphere using projected gradient descent."""
        c = project_to_sphere(c_init.copy())
        R, grad = compute_R_and_grad(c)
        best_local = c.copy()
        best_R_local = R
        
        # Adaptive step size
        step = 0.5
        momentum = 0.0
        v = np.zeros_like(c)
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Project gradient to tangent space
            grad_tangent = grad - np.dot(grad, c) * c
            
            # Momentum update (Polyak)
            v = 0.9 * v - step * grad_tangent
            # Project v to tangent space
            v = v - np.dot(v, c) * c
            
            # Trial step
            c_trial = c + v
            c_trial = project_to_sphere(c_trial)
            
            R_trial, grad_trial = compute_R_and_grad(c_trial)
            
            if R_trial < R:
                c = c_trial
                R = R_trial
                grad = grad_trial
                step = min(step * 1.05, 1.0) # Increase step slightly
                if R < best_R_local:
                    best_R_local = R
                    best_local = c.copy()
            else:
                # Backtracking
                step *= 0.5
                v = 0.0 # Reset momentum on failure
                if step < 1e-8:
                    break
            
            # Convergence check
            if it > 100 and abs(R - best_R_local) < tol:
                break
        
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
    
    # Add some structured guesses: symmetric pairs
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0 / np.sqrt(2)
            c_pair[j] = 1.0 / np.sqrt(2)
            initial_guesses.append(c_pair)
            
            c_pair_anti = np.zeros(n_coeffs)
            c_pair_anti[i] = 1.0 / np.sqrt(2)
            c_pair_anti[j] = -1.0 / np.sqrt(2)
            initial_guesses.append(c_pair_anti)

    np.random.seed(42)
    n_random = 50
    for _ in range(n_random):
        c_rand = np.random.randn(n_coeffs)
        c_rand = project_to_sphere(c_rand)
        initial_guesses.append(c_rand)
    
    # Run gradient descent on all guesses
    candidates = []
    for idx, c_init in enumerate(initial_guesses):
        if time.time() > deadline - 60:
            break
        
        c_opt, R_opt = projected_gradient_descent(c_init, max_iter=500, tol=1e-12)
        candidates.append((R_opt, c_opt))
        
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()
            
    # Sort candidates by R
    candidates.sort(key=lambda x: x[0])
    
    # Polish top candidates with SLSQP
    top_k = min(10, len(candidates))
    for i in range(top_k):
        if time.time() > deadline - 30:
            break
        
        c_init = candidates[i][1]
        
        constraints = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0}]
        
        def obj(c):
            return compute_R_and_grad(c)[0]
        
        def jac(c):
            return compute_R_and_grad(c)[1]
        
        try:
            res = minimize(obj, c_init, jac=jac, method='SLSQP',
                           constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-15, 'maxls': 50})
            
            if res.success and res.fun < best_R:
                best_R = res.fun
                best_c = res.x
        except Exception:
            pass

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
