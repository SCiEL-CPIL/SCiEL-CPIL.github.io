"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    
    Uses a derivative-free Levenberg-Marquardt (LM) algorithm with:
    - Rank-1 updated Jacobian approximation
    - Adaptive damping factor based on residual reduction
    - Deadline guard for safety
    
    Strategy:
    1. Start with x0
    2. Build initial Jacobian via finite differences (3 evals)
    3. LM iterations: 1 eval per iteration, rank-1 update Jacobian
    4. Adaptive damping: increase lambda if step fails, decrease if succeeds
    5. Final polish with tighter tolerance
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float, copy=True)
    
    # Helper to check time
    def time_ok():
        return time.time() < deadline
    
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x
    Fx = np.asarray(Fx, dtype=float)
    res_norm_sq = np.dot(Fx, Fx)
    res_norm = np.sqrt(res_norm_sq)
    
    if res_norm <= 1e-8:
        return x
    
    # Build initial Jacobian via central differences
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        xp = np.array(x, copy=True)
        xm = np.array(x, copy=True)
        xp[j] += h
        xm[j] -= h
        Fxp = F(xp)
        Fxm = F(xm)
        if np.all(np.isfinite(Fxp)) and np.all(np.isfinite(Fxm)):
            J[:, j] = (Fxp - Fxm) / (2 * h)
        else:
            # Fallback to forward difference
            Fxp = F(xp)
            if np.all(np.isfinite(Fxp)):
                J[:, j] = (Fxp - Fx) / h
            else:
                J[:, j] = 0.0
    
    if not np.all(np.isfinite(J)):
        J = np.eye(2)
    
    # LM parameters
    lam = 1e-3
    lam_min = 1e-12
    lam_max = 1e12
    best_x = x.copy()
    best_norm = res_norm
    best_Fx = Fx.copy()
    
    max_iter = 200
    stagnation_count = 0
    stagnation_threshold = 5
    
    for it in range(max_iter):
        if not time_ok():
            break
        
        # Compute LM step: (J^T J + lam * diag(J^T J)) * delta = -J^T F
        JTJ = J.T @ J
        diag_JTJ = np.diag(JTJ).copy()
        # Avoid zero diagonal
        diag_JTJ[diag_JTJ < 1e-15] = 1e-15
        
        A = JTJ + lam * np.diag(diag_JTJ)
        b = -J.T @ Fx
        
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lam *= 10
            if lam > lam_max:
                break
            continue
        
        if not np.all(np.isfinite(delta)):
            lam *= 10
            if lam > lam_max:
                break
            continue
        
        # Propose new point
        x_new = x + delta
        Fx_new = F(x_new)
        
        if not np.all(np.isfinite(Fx_new)):
            lam *= 10
            if lam > lam_max:
                break
            continue
        
        Fx_new = np.asarray(Fx_new, dtype=float)
        res_norm_sq_new = np.dot(Fx_new, Fx_new)
        res_norm_new = np.sqrt(res_norm_sq_new)
        
        # Check improvement
        if res_norm_sq_new < res_norm_sq:
            # Successful step
            # Update Jacobian via rank-1
            dx = x_new - x
            dF = Fx_new - Fx
            # Rank-1 update: J_new = J + (dF - J*dx) * dx^T / (dx^T dx)
            dx_norm_sq = np.dot(dx, dx)
            if dx_norm_sq > 1e-30:
                correction = (dF - J @ dx) / dx_norm_sq
                J = J + np.outer(correction, dx)
            
            x = x_new
            Fx = Fx_new
            res_norm_sq = res_norm_sq_new
            res_norm = res_norm_new
            
            # Track best
            if res_norm < best_norm:
                best_x = x.copy()
                best_norm = res_norm
                best_Fx = Fx.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Decrease lambda (step was good)
            lam = max(lam * 0.3, lam_min)
            
            # Check convergence
            if res_norm <= 1e-10:
                break
        else:
            # Unsuccessful step
            lam *= 10
            if lam > lam_max:
                break
            stagnation_count += 1
        
        if stagnation_count >= stagnation_threshold:
            # Try a small random perturbation to escape
            if time_ok():
                perturbation = 1e-3 * np.random.randn(2)
                x_try = x + perturbation
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    Fx_try = np.asarray(Fx_try, dtype=float)
                    res_try = np.linalg.norm(Fx_try)
                    if res_try < res_norm:
                        x = x_try
                        Fx = Fx_try
                        res_norm_sq = np.dot(Fx, Fx)
                        res_norm = res_try
                        # Rebuild Jacobian
                        for j in range(2):
                            xp = np.array(x, copy=True)
                            xp[j] += h
                            Fxp = F(xp)
                            if np.all(np.isfinite(Fxp)):
                                J[:, j] = (Fxp - Fx) / h
                        lam = 1e-3
                        stagnation_count = 0
                    else:
                        break
                else:
                    break
    
    # Final polish: if we're close, do a few more Newton steps with tighter tolerance
    if best_norm <= 1e-6 and time_ok():
        x = best_x.copy()
        Fx = best_Fx.copy()
        for _ in range(20):
            if not time_ok():
                break
            if np.linalg.norm(Fx) <= 1e-10:
                break
            # Rebuild Jacobian with smaller h
            h_small = 1e-8
            J_small = np.zeros((2, 2))
            for j in range(2):
                xp = np.array(x, copy=True)
                xp[j] += h_small
                Fxp = F(xp)
                if np.all(np.isfinite(Fxp)):
                    J_small[:, j] = (Fxp - Fx) / h_small
                else:
                    J_small[:, j] = J[:, j]
            
            if not np.all(np.isfinite(J_small)):
                break
            
            try:
                delta = np.linalg.solve(J_small, Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(delta)):
                break
            
            x_new = x - delta
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            Fx_new = np.asarray(Fx_new, dtype=float)
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                x = x_new
                Fx = Fx_new
            else:
                break
    
    # Return the best solution found
    if best_norm <= 1e-8:
        return best_x
    elif np.linalg.norm(best_Fx) <= 1e-8:
        return best_x
    else:
        # Return x if it's better
        if np.linalg.norm(Fx) < best_norm:
            return x
        return best_x

# EVOLVE-BLOCK-END
