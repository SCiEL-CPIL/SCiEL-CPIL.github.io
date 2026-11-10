import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import time

def rudin_problem10() -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    l = 10
    N_theta = 70
    N_phi = 130
    
    # 1. Quadrature Points
    x_nodes, x_weights = sp.roots_legendre(N_theta)
    phi_nodes = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    phi_weights = np.full(N_phi, 2 * np.pi / N_phi)
    
    x_grid, phi_grid = np.meshgrid(x_nodes, phi_nodes, indexing='ij')
    w_grid = np.outer(x_weights, phi_weights)
    
    x_flat = x_grid.flatten()
    phi_flat = phi_grid.flatten()
    w_flat = w_grid.flatten()
    
    N_points = len(x_flat)
    
    # 2. Precompute Basis Matrix A (N_points x 21)
    m_vals = np.arange(-l, l + 1)
    A = np.zeros((N_points, 2 * l + 1))
    
    P_vals = {}
    for m in range(l + 1):
        P_vals[m] = sp.lpmv(m, l, x_flat)
        
    N_norm = {}
    for m in range(l + 1):
        log_fact_ratio = sp.gammaln(l - m + 1) - sp.gammaln(l + m + 1)
        N_norm[m] = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(log_fact_ratio))
        
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        norm = N_norm[abs_m]
        P_m = P_vals[abs_m]
        
        if m == 0:
            A[:, idx] = norm * P_m
        elif m > 0:
            A[:, idx] = np.sqrt(2) * norm * P_m * np.cos(m * phi_flat)
        else:
            A[:, idx] = np.sqrt(2) * norm * P_m * np.sin(abs_m * phi_flat)
            
    def objective(c):
        f = A @ c
        f2 = f * f
        f4 = f2 * f2
        S2 = np.dot(w_flat, f2)
        S4 = np.dot(w_flat, f4)
        
        if S2 < 1e-12:
            return 1e12
            
        R4 = S4 / (S2 * S2)
        return R4

    def objective_grad(c):
        f = A @ c
        f2 = f * f
        f4 = f2 * f2
        S2 = np.dot(w_flat, f2)
        S4 = np.dot(w_flat, f4)
        
        if S2 < 1e-12:
            return np.zeros_like(c)
            
        dS2_dc = 2 * (A.T @ (w_flat * f))
        dS4_dc = 4 * (A.T @ (w_flat * f * f2))
        
        grad = (1.0 / (S2 * S2)) * dS4_dc - (2.0 * S4 / (S2 ** 3)) * dS2_dc
        return grad

    # Identify indices for even m modes: 0, ±2, ±4, ±6, ±8, ±10
    # m: -10, -9, ..., 0, ..., 10
    # Index i corresponds to m = i - 10
    # m=0 -> idx 10
    # m=±2 -> idx 8, 12
    # m=±4 -> idx 6, 14
    # m=±6 -> idx 4, 16
    # m=±8 -> idx 2, 18
    # m=±10 -> idx 0, 20
    
    even_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # We will optimize over these 11 coefficients
    # To ensure we find the global min in this subspace, we use multiple restarts
    
    best_c_full = np.zeros(21)
    best_score = np.inf
    
    # Initial guess: pure m=0
    c0_guess = np.zeros(21)
    c0_guess[10] = 1.0
    score0 = objective(c0_guess)
    if score0 < best_score:
        best_score = score0
        best_c_full = c0_guess.copy()
        
    # Initial guess: pure m=10
    c10_guess = np.zeros(21)
    c10_guess[20] = 1.0
    score10 = objective(c10_guess)
    if score10 < best_score:
        best_score = score10
        best_c_full = c10_guess.copy()

    # Optimize in the reduced subspace
    # Map reduced c_reduced (11) to full c (21)
    def reduced_to_full(c_red):
        c_full = np.zeros(21)
        for i, idx in enumerate(even_indices):
            c_full[idx] = c_red[i]
        return c_full

    def objective_reduced(c_red):
        return objective(reduced_to_full(c_red))

    def objective_grad_reduced(c_red):
        c_full = reduced_to_full(c_red)
        grad_full = objective_grad(c_full)
        grad_red = np.zeros(11)
        for i, idx in enumerate(even_indices):
            grad_red[i] = grad_full[idx]
        return grad_red

    np.random.seed(42)
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > time_limit - safety_margin:
            break
            
        # Random start in reduced space
        c_red0 = np.random.randn(11)
        c_red0 /= np.linalg.norm(c_red0)
        
        try:
            res = opt.minimize(objective_reduced, c_red0, jac=objective_grad_reduced, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                c_red_opt = res.x
                c_full_opt = reduced_to_full(c_red_opt)
                score_opt = objective(c_full_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_c_full = c_full_opt.copy()
        except Exception as e:
            pass

    # Final polish on best found
    if best_score < 1e10:
        # Project best_c_full back to reduced space if it's close to it?
        # Or just polish in full space?
        # Polishing in full space might drift out of the "flattest" family if the global min is actually non-axisymmetric.
        # But the directive says to exploit symmetry.
        # Let's polish in the reduced space.
        
        # Extract reduced coefficients from best_c_full
        c_red_best = np.zeros(11)
        for i, idx in enumerate(even_indices):
            c_red_best[i] = best_c_full[idx]
            
        try:
            res_final = opt.minimize(objective_reduced, c_red_best, jac=objective_grad_reduced, method='L-BFGS-B', 
                                     options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res_final.success:
                c_red_final = res_final.x
                c_full_final = reduced_to_full(c_red_final)
                score_final = objective(c_full_final)
                if score_final < best_score:
                    best_score = score_final
                    best_c_full = c_full_final.copy()
        except Exception as e:
            pass

    # Ensure validity: max|c| in [1e-3, 1e5]
    max_c = np.max(np.abs(best_c_full))
    if max_c < 1e-3:
        best_c_full *= (1.0 / max_c)
    elif max_c > 1e5:
        best_c_full *= (1e5 / max_c)

    return best_c_full
