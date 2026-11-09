import numpy as np
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Random points on the sphere for the surrogate objective
    rng = np.random.default_rng(42)
    M = 5000
    theta = rng.uniform(0, 2 * np.pi, M)
    phi_rand = np.arccos(2 * rng.uniform(0, 1, M) - 1)
    Y = np.column_stack([
        np.sin(phi_rand) * np.cos(theta),
        np.sin(phi_rand) * np.sin(theta),
        np.cos(phi_rand)
    ])
    
    beta = 10.0  # Sharpness of the soft-max
    
    def objective_and_grad(P_flat):
        P = P_flat.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        U = P / norms
        
        diff = Y[:, np.newaxis, :] - U[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2) # (M, N)
        
        d2_scaled = beta * d2
        max_d2 = np.max(d2_scaled, axis=1, keepdims=True)
        exp_d2 = np.exp(d2_scaled - max_d2)
        sum_exp_d2 = np.sum(exp_d2, axis=1, keepdims=True)
        w = exp_d2 / sum_exp_d2 # (M, N)
        
        log_sum = max_d2 + np.log(sum_exp_d2)
        F = np.sum(log_sum) / beta
        
        diff_full = U[np.newaxis, :, :] - Y[:, np.newaxis, :] # (M, N, 3)
        grad_U = 2.0 * beta * np.sum(w[:, :, np.newaxis] * diff_full, axis=0) # (N, 3)
        
        dot = np.sum(grad_U * U, axis=1, keepdims=True) # (N, 1)
        grad_tangent = grad_U - dot * U # (N, 3)
        grad_P = grad_tangent / norms # (N, 3)
        
        return F, grad_P.flatten()
    
    best_P = P_init
    best_F = np.inf
    
    # Run optimization
    res = minimize(
        objective_and_grad,
        P_init.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
    )
    
    if res.success or res.status == 0:
        P_opt = res.x.reshape(N, 3)
        P_opt = P_opt / np.linalg.norm(P_opt, axis=1, keepdims=True)
        
        # Evaluate objective for comparison
        F_opt, _ = objective_and_grad(P_opt.flatten())
        if F_opt < best_F:
            best_F = F_opt
            best_P = P_opt
    
    # If time remains, try a second run with different beta or random init?
    # For now, return the best found.
    
    return best_P
