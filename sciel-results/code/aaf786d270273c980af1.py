"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Strategy: Global Homotopy Path Following with Adaptive Step Control
    # 1. Define a homotopy H(t, x) = (1-t)*F(x) + t*(x - x_target) where x_target is a simple anchor (e.g., 0 or x0).
    # 2. Solve for x(t) starting from t=0 (where x=x_target is an exact root) and marching to t=1.
    # 3. Use a predictor-corrector scheme: 
    #    - Predictor: Linear extrapolation of the path.
    #    - Corrector: Damped Newton steps on H(t, x) = 0.
    # 4. This provides a global convergence mechanism that is robust to the initial guess x0,
    #    effectively "pulling" the solution from a known root to the target system's root.
    # 5. If path following fails, fallback to local Newton with restarts.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False

    def newton_damped(F, x0, max_iter=30, tol=1e-12):
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
            
            try:
                Fx = F(x)
            except Exception:
                break
            
            if not np.all(np.isfinite(Fx)):
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            if current_norm <= tol:
                return x
            
            # Compute Jacobian via central differences for better accuracy
            eps = 1e-6
            J = np.zeros((2, 2))
            valid = True
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                try:
                    J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
                except Exception:
                    valid = False
                    break
            
            if not valid or not np.all(np.isfinite(J)):
                break
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Damped Newton Step (Line Search)
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue
                
                if not np.all(np.isfinite(F_new)):
                    continue
                
                new_norm = norm(F_new)
                if new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break
            
            if not step_found:
                break
                
            if norm(dx) < 1e-10:
                break
        
        return best_x

    def homotopy_solver(F, x0):
        # Anchor point: We use x0 as the anchor for the "simple" system.
        # Simple system: G(x) = x - x0 = 0. Root is x0.
        # Homotopy: H(t, x) = (1-t)*F(x) + t*(x - x0) = 0
        # At t=0: F(x) = 0 (Target)
        # At t=1: x - x0 = 0 => x = x0 (Known)
        
        # We march from t=1 to t=0.
        # x(1) = x0
        # We want to find x(0).
        
        x = np.array(x0, dtype=float)
        t = 1.0
        
        # Adaptive step size in t
        dt = 0.1
        min_dt = 1e-4
        
        for _ in range(200):  # Max steps
            if time_left() < 1.0:
                break
            if t <= 0:
                break
            
            # Predictor: Linear extrapolation
            # x(t-dt) approx x(t) - dt * (dx/dt)
            # We don't have dx/dt, so we just try a step and correct.
            # Simple predictor: x_pred = x
            # Better predictor: Use previous step direction if available.
            # For simplicity, let's use x_pred = x.
            
            t_new = max(0.0, t - dt)
            
            # Define the homotopy function at t_new
            def H(x, t_val):
                return (1 - t_val) * F(x) + t_val * (x - x0)
            
            def J_H(x, t_val):
                # Jacobian of H w.r.t x
                # H = (1-t)F(x) + t(x-x0)
                # dH/dx = (1-t) J_F(x) + t I
                eps = 1e-6
                J_F = np.zeros((2, 2))
                valid = True
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = eps
                    try:
                        J_F[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
                    except Exception:
                        valid = False
                        break
                if not valid:
                    return None
                return (1 - t_val) * J_F + t_val * np.eye(2)
            
            # Corrector: Newton on H(x, t_new) = 0
            x_curr = x.copy()
            converged = False
            
            for _iter in range(10):
                if time_left() < 0.5:
                    break
                
                try:
                    H_val = H(x_curr, t_new)
                except Exception:
                    break
                
                if not np.all(np.isfinite(H_val)):
                    break
                
                if norm(H_val) < 1e-10:
                    converged = True
                    break
                
                J_val = J_H(x_curr, t_new)
                if J_val is None or not np.all(np.isfinite(J_val)):
                    break
                
                try:
                    dx = np.linalg.solve(J_val, -H_val)
                except np.linalg.LinAlgError:
                    break
                
                if not np.all(np.isfinite(dx)):
                    break
                
                # Damped step
                step_found = False
                for lam in [1.0, 0.5, 0.25]:
                    x_try = x_curr + lam * dx
                    try:
                        H_try = H(x_try, t_new)
                    except Exception:
                        continue
                    if np.all(np.isfinite(H_try)) and norm(H_try) < norm(H_val):
                        x_curr = x_try
                        step_found = True
                        break
                
                if not step_found:
                    break
            
            if converged:
                x = x_curr
                t = t_new
                dt = min(dt * 1.5, 0.2) # Increase step if successful
            else:
                # Reduce step size
                dt = max(dt * 0.5, min_dt)
                if dt == min_dt:
                    # Stuck, break
                    break
        
        return x

    # Phase 1: Try Homotopy from x0
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    x_hom = homotopy_solver(F, x0)
    
    if is_good(x_hom):
        return x_hom
    
    # Phase 2: Try Newton from the result of Homotopy (if it got close)
    x_newton = newton_damped(F, x_hom)
    if is_good(x_newton):
        return x_newton
    
    # Phase 3: Fallback to Random Restarts + Newton
    scale = max(1.0, norm(x0))
    rng = np.random.default_rng(42)
    
    candidates = [x0, x_hom, np.array([0.0, 0.0])]
    
    for r in [scale * 0.1, scale * 0.5, scale * 1.0]:
        for _ in range(5):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)
            
    best_x = x_newton
    best_norm = np.inf
    try:
        best_norm = norm(F(best_x))
    except Exception:
        pass
    
    for c in candidates:
        if time_left() < 1.0:
            break
        if not np.all(np.isfinite(c)):
            continue
        
        x_try = newton_damped(F, c)
        try:
            F_try = F(x_try)
            if np.all(np.isfinite(F_try)):
                n_try = norm(F_try)
                if n_try < best_norm:
                    best_norm = n_try
                    best_x = x_try
                if n_try <= 1e-8:
                    return x_try
        except Exception:
            pass
            
    return best_x
# EVOLVE-BLOCK-END
