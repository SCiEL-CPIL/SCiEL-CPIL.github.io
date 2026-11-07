"""Root finder for 3 fixed 2D systems using homotopy/continuation with Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses homotopy/continuation method to track a root path from x0,
    then refines with Broyden's method and a high-precision polish.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    tol = 1e-8
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))

    if best_norm <= tol:
        return best_x

    # Helper: compute finite-difference Jacobian (costs 2 F calls, reuses Fx)
    def fd_jacobian(x, Fx, h=1e-5):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = F(x + e)
            J[:, j] = (Fx_pert - Fx) / h
        return J

    # Define the homotopy function F_t(x) = t*F(x) + (1-t)*(x - x0)
    def F_t(x, t):
        return t * F(x) + (1 - t) * (x - x)

    # Broyden's method on a local basin
    def broyden_solve(x_start, max_iters=50):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        # Initial Jacobian via finite differences (2 F calls)
        J = fd_jacobian(x_local, Fx)

        for i in range(max_iters):
            if time_left() < 2.0:
                break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-8 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Backtracking line search with any-decrease acceptance
            alpha = 1.0
            accepted = False
            for _ in range(10):
                if time_left() < 1.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx * (1 + 1e-4):  # Accept any decrease
                    # Update Jacobian (Broyden rank-1)
                    s = alpha * dx
                    y = F_try - Fx
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-14:
                        J += np.outer(y - Js, s) / denom
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    accepted = True
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    if norm_Fx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local

    # Final polish pass: high-precision Newton-like steps when close to root
    def polish(x_start, max_iters=20):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        polish_tol = 1e-10
        
        for i in range(max_iters):
            if time_left() < 1.0:
                break
            
            # Compute Jacobian via finite differences
            J = fd_jacobian(x_local, Fx, h=1e-6)
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-10 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Damped Newton step
            alpha = 1.0
            for _ in range(6):
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    if norm_Fx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

        return x_local

    # Homotopy/continuation method
    # Solve F_t(x) = 0 for t from 0 to 1
    # At t=0: F_0(x) = x - x0, so root is x0
    # At t=1: F_1(x) = F(x), so root is the desired solution
    def homotopy_solve(x_start, max_steps=100):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        t = 0.0
        dt = 0.05  # Initial step size in t
        
        for step in range(max_steps):
            if time_left() < 3.0:
                break
            if t >= 1.0:
                break
            
            # Try to increase t by dt
            t_new = min(1.0, t + dt)
            
            # Solve F_t_new(x) = 0 starting from current x_local
            # Use a few Newton iterations for the homotopy equation
            x_hom = x_local.copy()
            converged = False
            
            for inner_iter in range(20):
                if time_left() < 2.0:
                    break
                
                # Compute F_t_new(x_hom)
                Fx_hom = t_new * F(x_hom) + (1 - t_new) * (x_hom - x)
                norm_Fx = np.linalg.norm(Fx_hom)
                
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_hom.copy()
                
                if norm_Fx <= 1e-10:
                    converged = True
                    break
                
                # Compute Jacobian of F_t_new
                # J_t = t * J_F + (1-t) * I
                # We approximate J_F via finite differences
                h = 1e-5
                J_F = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_pert = t_new * F(x_hom + e) + (1 - t_new) * (x_hom + e - x)
                    J_F[:, j] = (Fx_pert - Fx_hom) / h
                
                # Actually, J_t = t * J_F + (1-t) * I
                # But we already computed J_F including the (1-t) part
                # Let's recompute properly
                J_t = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_pert = t_new * F(x_hom + e) + (1 - t_new) * (x_hom + e - x)
                    J_t[:, j] = (Fx_pert - Fx_hom) / h
                
                try:
                    dx = np.linalg.solve(J_t, -Fx_hom)
                except np.linalg.LinAlgError:
                    try:
                        J_reg = J_t + 1e-8 * np.eye(2)
                        dx = np.linalg.solve(J_reg, -Fx_hom)
                    except np.linalg.LinAlgError:
                        break
                
                # Damped step
                alpha = 1.0
                accepted = False
                for _ in range(5):
                    x_try = x_hom + alpha * dx
                    Fx_try = t_new * F(x_try) + (1 - t_new) * (x_try - x)
                    norm_try = np.linalg.norm(Fx_try)
                    if norm_try < norm_Fx * 0.9:
                        x_hom = x_try
                        accepted = True
                        break
                    else:
                        alpha *= 0.5
                
                if not accepted:
                    break
            
            if converged:
                x_local = x_hom.copy()
                t = t_new
                # Increase dt slightly for faster progress
                dt = min(0.2, dt * 1.2)
            else:
                # Reduce dt and try again
                dt *= 0.5
                if dt < 1e-4:
                    break
        
        return x_local

    # Main strategy:
    # 1. Try homotopy from x0
    # 2. If that doesn't converge well, try Broyden from x0
    # 3. If still not converged, try restarts
    
    # First, try homotopy method
    x_hom = homotopy_solve(x)
    
    # Check if homotopy gave us a good starting point
    Fx_hom = F(x_hom)
    norm_hom = np.linalg.norm(Fx_hom)
    if norm_hom < best_norm:
        best_norm = norm_hom
        best_x = x_hom.copy()
    
    # Now refine with Broyden from the homotopy result
    if time_left() > 5.0:
        broyden_solve(x_hom)
    
    # If we're close, polish
    if best_norm < 1e-4 and time_left() > 3.0:
        polish(best_x)
    
    if best_norm <= tol:
        return best_x
    
    # If homotopy didn't work well, try Broyden directly from x0
    if time_left() > 5.0:
        broyden_solve(x)
    
    if best_norm <= tol:
        return best_x
    
    if best_norm < 1e-4 and time_left() > 3.0:
        polish(best_x)
    
    if best_norm <= tol:
        return best_x
    
    # Restart strategy if still not converged
    n_restarts = 0
    max_restarts = 20
    
    while n_restarts < max_restarts and time_left() > 5.0:
        if best_norm <= tol:
            return best_x
        
        # Choose new starting point
        if n_restarts % 3 == 0:
            current_x = x + np.random.normal(0, 1.0, size=2)
        elif n_restarts % 3 == 1:
            radius = min(10.0, 0.5 * (n_restarts + 1))
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec
        else:
            current_x = x + np.random.uniform(-5, 5, size=2)
        
        broyden_solve(current_x)
        
        if best_norm < 1e-4 and time_left() > 3.0:
            polish(best_x)
        
        if best_norm <= tol:
            return best_x
        
        n_restarts += 1
    
    # Final polish before returning
    if best_norm < 1e-3 and time_left() > 2.0:
        polish(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
