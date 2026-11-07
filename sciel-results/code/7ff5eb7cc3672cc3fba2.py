"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts and polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses Broyden's quasi-Newton method with rank-1 Jacobian updates,
    Levenberg-Marquardt regularization, and adaptive restarts.
    """
    x = np.array(x0, dtype=float)
    tol = 1e-8
    deadline = time.time() + 100.0  # Safety margin for 120s limit
    call_budget = 480  # Stay well under 500
    
    # Call counter
    calls = [0]
    
    def F_counted(x):
        calls[0] += 1
        return np.asarray(F(x), dtype=float)
    
    # Initial evaluation
    Fx = F_counted(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= tol:
        return best_x
    
    # Initialize Jacobian with identity (will be updated by Broyden)
    J = np.eye(2)
    
    max_restarts = 12
    max_local_iters = 60
    
    for restart_iter in range(max_restarts):
        if time.time() > deadline - 5:
            break
        if calls[0] >= call_budget:
            break
        
        x_local = best_x.copy() if restart_iter > 0 else x.copy()
        Fx_local = F_counted(x_local)
        norm_Fx = np.linalg.norm(Fx_local)
        
        if norm_Fx <= tol:
            return x_local
        
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        
        # Reset Jacobian for each restart (fresh start)
        J = np.eye(2)
        
        # LM damping parameter
        lam = 1e-6
        stagnation = 0
        
        for i in range(max_local_iters):
            if time.time() > deadline - 1:
                return best_x
            if calls[0] >= call_budget:
                return best_x
            
            # Solve (J + lam*I) dx = -Fx for Newton-like step
            try:
                A = J + lam * np.eye(2)
                dx = np.linalg.solve(A, -Fx_local)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse or random step
                try:
                    dx = np.linalg.lstsq(J, -Fx_local, rcond=None)[0]
                except:
                    dx = np.random.randn(2) * 0.1
                    lam *= 10
                    continue
            
            # Limit step size
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx / step_norm * 10.0
            
            # Try the step
            x_try = x_local + dx
            Fx_try = F_counted(x_try)
            norm_try = np.linalg.norm(Fx_try)
            
            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()
            
            if norm_try <= tol:
                return x_try
            
            if norm_try < norm_Fx * 0.99:
                # Accept step
                s = x_try - x_local
                y = Fx_try - Fx_local
                
                # Broyden rank-1 update: J_new = J + (y - J*s) * s^T / (s^T * s)
                sTs = np.dot(s, s)
                if sTs > 1e-12:
                    J = J + np.outer(y - J @ s, s) / sTs
                
                # Reduce damping on success
                lam = max(lam * 0.5, 1e-10)
                stagnation = 0
                
                x_local = x_try
                Fx_local = Fx_try
                norm_Fx = norm_try
            else:
                # Reject step, increase damping
                lam *= 5.0
                stagnation += 1
                
                if stagnation > 8:
                    break
        
        # Check if we converged
        if best_norm <= tol:
            return best_x
        
        # Prepare restart point
        if restart_iter < 4:
            # Spread out from x0
            radius = 1.0 + restart_iter * 0.5
            angle = restart_iter * 1.3 + 0.5
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_local = x + offset
        elif restart_iter < 8:
            # Spread out from best_x
            radius = 0.5 + (restart_iter - 4) * 0.3
            angle = (restart_iter - 4) * 1.7 + 0.3
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_local = best_x + offset
        else:
            # Wider exploration
            radius = 1.0 + (restart_iter - 8) * 0.5
            angle = (restart_iter - 8) * 2.1 + 0.7
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_local = best_x + offset
        
        # Evaluate at new restart point
        Fx_check = F_counted(x_local)
        norm_check = np.linalg.norm(Fx_check)
        if norm_check < best_norm:
            best_norm = norm_check
            best_x = x_local.copy()
    
    # Final polish: try a few more iterations at best_x with fresh Jacobian
    if calls[0] < call_budget - 5:
        x_local = best_x.copy()
        Fx_local = F_counted(x_local)
        norm_Fx = np.linalg.norm(Fx_local)
        
        if norm_Fx <= tol:
            return x_local
        
        # Use finite difference for a final Jacobian estimate if we have budget
        if calls[0] < call_budget - 3:
            h = 1e-6 * max(1.0, np.linalg.norm(x_local))
            J_fd = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = F_counted(x_local + e)
                J_fd[:, j] = (Fx_pert - Fx_local) / h
            
            # A few damped Newton steps with this Jacobian
            for _ in range(5):
                if calls[0] >= call_budget:
                    break
                try:
                    dx = np.linalg.solve(J_fd, -Fx_local)
                except:
                    break
                step_norm = np.linalg.norm(dx)
                if step_norm > 5.0:
                    dx = dx / step_norm * 5.0
                x_try = x_local + dx
                Fx_try = F_counted(x_try)
                norm_try = np.linalg.norm(Fx_try)
                if norm_try < norm_Fx:
                    x_local = x_try
                    Fx_local = Fx_try
                    norm_Fx = norm_try
                    if norm_try < best_norm:
                        best_norm = norm_try
                        best_x = x_try.copy()
                    if norm_try <= tol:
                        return x_try
                else:
                    break
    
    return best_x
# EVOLVE-BLOCK-END
