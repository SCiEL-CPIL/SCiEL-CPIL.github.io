import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Test Points for Covering Radius Estimation
    M = 5000
    j = np.arange(M) + 0.5
    z_t = 1.0 - 2.0 * j / M
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(1.0 - z_t * z_t)
    Y = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    # Pre-compute Y norms (should be 1)
    # Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    def objective_and_grad(P_flat):
        P = P_flat.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P = P / norms
        
        dots = Y @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        min_angles = np.min(angles, axis=1)
        
        lambda_sm = 50.0
        max_angle = np.max(min_angles)
        exps = np.exp(lambda_sm * (min_angles - max_angle))
        sum_exps = np.sum(exps)
        
        if sum_exps == 0:
             return 0.0, np.zeros_like(P_flat)
             
        loss = (1.0 / lambda_sm) * (np.log(sum_exps) + lambda_sm * max_angle)
        
        weights = exps / sum_exps
        
        nearest_idx = np.argmin(angles, axis=1)
        
        dots_nearest = dots[np.arange(M), nearest_idx]
        dots_nearest = np.clip(dots_nearest, -1.0, 1.0)
        denom = np.sqrt(np.maximum(0.0, 1.0 - dots_nearest**2))
        denom[denom < 1e-8] = 1e-8
        
        grad_contrib = weights[:, None] * (-Y / denom[:, None])
        
        grad_P = np.zeros_like(P)
        np.add.at(grad_P, nearest_idx, grad_contrib)
        
        return loss, grad_P.reshape(-1)

    P_current = P.copy()
    best_loss, _ = objective_and_grad(P_current.flatten())
    best_P = P_current.copy()
    
    alpha = 0.1
    min_alpha = 1e-6
    max_iter = 1000
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        loss, grad = objective_and_grad(P_current.flatten())
        
        if loss < best_loss:
            best_loss = loss
            best_P = P_current.copy()
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-7:
            break
            
        # Backtracking line search
        step_dir = -grad / grad_norm
        a = alpha
        loss_new = np.inf
        P_try = P_current.copy()
        
        for _ in range(10):
            P_try = P_current + a * step_dir
            # Project to sphere
            norms = np.linalg.norm(P_try, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P_try = P_try / norms
            
            loss_new, _ = objective_and_grad(P_try.flatten())
            
            if loss_new < loss:
                break
            a *= 0.5
            
        P_current = P_try
        alpha = max(a, min_alpha)
        
        # If we couldn't find a good step, reduce alpha
        if loss_new >= loss:
             alpha *= 0.5

    # Final polish
    if time.time() < deadline - 5.0:
        P_current = best_P.copy()
        alpha = 0.01
        for it in range(200):
            if time.time() > deadline - 2.0:
                break
            loss, grad = objective_and_grad(P_current.flatten())
            if loss < best_loss:
                best_loss = loss
                best_P = P_current.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-7:
                break
                
            step_dir = -grad / grad_norm
            a = alpha
            P_try = P_current.copy()
            loss_new = np.inf
            
            for _ in range(10):
                P_try = P_current + a * step_dir
                norms = np.linalg.norm(P_try, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                P_try = P_try / norms
                loss_new, _ = objective_and_grad(P_try.flatten())
                if loss_new < loss:
                    break
                a *= 0.5
                
            P_current = P_try

    return best_P
