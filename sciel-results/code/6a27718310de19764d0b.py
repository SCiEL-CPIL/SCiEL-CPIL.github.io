import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize
import time
import math

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    k = 10
    n_theta = 70
    n_phi = 130
    n_points = n_theta * n_phi
    
    # 1. Precompute quadrature nodes and weights
    # Gauss-Legendre in mu = cos(theta)
    mu, w_theta = np.polynomial.legendre.leggauss(n_theta)
    
    # Equally spaced phi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    
    # Meshgrid for quadrature points
    # We want a flat array of points
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Weights
    w_theta_rep = np.repeat(w_theta, n_phi)
    w_flat = w_theta_rep * w_phi
    
    # 2. Precompute Basis Functions Y_m
    # m ranges from -10 to 10
    m_vals = np.arange(-k, k + 1)
    Y_basis = np.zeros((n_points, 2 * k + 1), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        
        # Associated Legendre P_10^abs_m(mu)
        # lpmv(m, l, x) returns P_l^m(x) with Condon-Shortley phase
        P_lm = lpmv(abs_m, k, mu_flat)
        
        # Normalization N_m
        # N_m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
        # With l=10: N_m = sqrt(21/(4pi) * (10-m)!/(10+m)!)
        fact_num = math.factorial(k - abs_m)
        fact_den = math.factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * fact_num / fact_den)
        
        # Wait, the prompt says:
        # Y_0 = N_0 P^0
        # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
        # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
        
        if m == 0:
            Y_basis[:, idx] = N_m * P_lm
        elif m > 0:
            Y_basis[:, idx] = np.sqrt(2) * N_m * P_lm * np.cos(m * phi_flat)
        else:
            Y_basis[:, idx] = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * phi_flat)
            
    # 3. Optimization
    def objective_and_grad(c):
        f = Y_basis @ c
        f2 = f * f
        f4 = f2 * f2
        
        Q2 = np.dot(w_flat, f2)
        Q4 = np.dot(w_flat, f4)
        
        if Q2 < 1e-12:
            return 1e10, np.zeros_like(c)
            
        J = Q4 / (Q2 * Q2)
        
        # Gradient
        # J = Q4 * Q2^-2
        # dJ/dc = (dQ4/dc) * Q2^-2 - 2 * Q4 * Q2^-3 * (dQ2/dc)
        # dQ4/dc = 4 * sum_p w_p f_p^3 * (df_p/dc) = 4 * Y^T (w * f^3)
        # dQ2/dc = 2 * sum_p w_p f_p * (df_p/dc) = 2 * Y^T (w * f)
        
        w_f3 = w_flat * (f * f2) # w * f^3
        w_f = w_flat * f
        
        grad_Q4 = 4.0 * (Y_basis.T @ w_f3)
        grad_Q2 = 2.0 * (Y_basis.T @ w_f)
        
        grad_J = (grad_Q4 / (Q2 * Q2)) - (2.0 * Q4 / (Q2 ** 3)) * grad_Q2
        
        return J, grad_J

    best_c = np.ones(2 * k + 1)
    best_val = np.inf
    
    # Random restarts with multiple strategies
    n_restarts = 30
    np.random.seed(42)
    
    # Strategy 1: Start from all-ones with tight tolerances
    c0 = np.ones(2 * k + 1)
    c0 /= np.linalg.norm(c0)
    try:
        res = minimize(
            objective_and_grad, 
            c0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        val, _ = objective_and_grad(res.x)
        if val < best_val:
            best_val = val
            best_c = res.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Random restarts
    for i in range(1, n_restarts):
        if time.time() > deadline - 120:  # Leave 120s for polish
            break
            
        c0 = np.random.randn(2 * k + 1)
        c0 /= np.linalg.norm(c0)
        
        try:
            res = minimize(
                objective_and_grad, 
                c0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            val, _ = objective_and_grad(res.x)
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 30:
        try:
            res_polish = minimize(
                objective_and_grad, 
                best_c, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            val, _ = objective_and_grad(res_polish.x)
            if val < best_val:
                best_val = val
                best_c = res_polish.x.copy()
        except Exception:
            pass
    
    # Ensure validity: max|c| between 1e-3 and 1e5
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c *= 1.0 / max_abs
    elif max_abs > 1e5:
        best_c *= 1e5 / max_abs
        
    return best_c
