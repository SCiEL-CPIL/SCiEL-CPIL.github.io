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
    
    # Precompute weighted M^T for fast gradient: (W * M) transposed
    # u = M @ (W * f^3) = (M * W[:,None])^T @ f^3 ... actually M @ (W * f^3)
    # Let MW = M * W[None, :] so MW @ f = M @ (W * f)
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
        
        # Gradient:
        # dA/dc = 4 * M @ (W * f^3) = 4 * MW @ f^3
        # dB/dc = 2 * M @ (W * f) = 2 * MW @ f
        u = MW @ (f ** 3)
        v = MW @ f
        
        grad_R = (1.0 * (norm4_4 ** -0.75) * (norm2_sq ** -0.5) * u) - \
                 (1.0 * (norm4_4 ** 0.25) * (norm2_sq ** -1.5) * v)
        
        return R, grad_R

    def trust_region_descent(c_init, max_iter=2000, tol=1e-12):
        """Minimize R on the unit sphere using a trust-region-like method with adaptive steps."""
        c = c_init.copy()
        c /= np.linalg.norm(c)
        
        R, grad = compute_R_and_grad(c)
        best_local = c.copy()
        best_R_local = R
        
        # Trust region radius
        delta = 1.0
        min_delta = 1e-6
        max_delta = 5.0
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Project gradient onto tangent space: grad - (grad·c)c
            grad_tangent = grad - np.dot(grad, c) * c
            g_norm = np.linalg.norm(grad_tangent)
            
            if g_norm < tol:
                break
            
            # Determine step direction: -grad_tangent
            direction = -grad_tangent / g_norm
            
            # Determine step size within trust region
            # Try several step sizes within [delta/10, delta]
            # We want to find a step that decreases R sufficiently
            best_step = 0.0
            best_R_step = R
            best_c_step = c
            
            # Candidate step sizes
            # Geometric progression from small to delta
            n_trials = 5
            steps = np.geomspace(max(delta * 0.1, 1e-8), delta, n_trials)
            
            for s in steps:
                c_trial = c + s * direction
                c_trial /= np.linalg.norm(c_trial)
                R_trial, _ = compute_R_and_grad(c_trial)
                
                if R_trial < best_R_step:
                    best_R_step = R_trial
                    best_step = s
                    best_c_step = c_trial
            
            # Update trust region
            if best_step > 0:
                # Success: expand trust region
                delta = min(delta * 1.5, max_delta)
                c = best_c_step
                R = best_R_step
                _, grad = compute_R_and_grad(c)
                
                if R < best_R_local:
                    best_R_local = R
                    best_local = c.copy()
            else:
                # Failure: shrink trust region
                delta = max(delta * 0.5, min_delta)
                if delta < min_delta:
                    break
        
        return best_local, best_R_local

    # Initial candidates
    c0 = np.zeros(n_coeffs)
    c0[10] = 1.0  # m=0
    R0, _ = compute_R_and_grad(c0)
    best_R = R0
    best_c = c0.copy()
    
    # Structured initial guesses
    initial_guesses = [c0.copy()]
    
    # Single m harmonics
    for m_idx in range(n_coeffs):
        c_single = np.zeros(n_coeffs)
        c_single[m_idx] = 1.0
        initial_guesses.append(c_single)
    
    # Pairs of harmonics
    for i in range(n_coeffs):
        for j in range(i + 1, n_coeffs):
            c_pair = np.zeros(n_coeffs)
            c_pair[i] = 1.0 / np.sqrt(2)
            c_pair[j] = 1.0 / np.sqrt(2)
            initial_guesses.append(c_pair)
    
    # Random restarts
    np.random.seed(42)
    n_random = 30
    for _ in range(n_random):
        c_rand = np.random.randn(n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    # Optimize each initial guess
    for idx, c_init in enumerate(initial_guesses):
        if time.time() > deadline - 40:
            break
        
        c_opt, R_opt = trust_region_descent(c_init, max_iter=1000, tol=1e-12)
        
        if R_opt < best_R:
            best_R = R_opt
            best_c = c_opt.copy()
    
    # Final polish: use SLSQP from the best point found
    if time.time() < deadline - 20:
        constraints = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0}]
        
        def obj(c):
            return compute_R_and_grad(c)[0]
        
        def jac(c):
            return compute_R_and_grad(c)[1]
        
        res = minimize(obj, best_c, jac=jac, method='SLSQP',
                       constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'maxls': 50})
        
        if res.fun < best_R:
            best_c = res.x
            best_R = res.fun
    
    # Ensure output is valid
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
