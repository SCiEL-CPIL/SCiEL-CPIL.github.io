"""Root finder for 3 fixed two-dimensional systems using Broyden's method with polish and Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy: 
    1. Broyden's method (1 F call/iter after 3-call setup) from x0 and restarts.
    2. Polish phase with finite-diff Newton when close to root.
    3. Nelder-Mead simplex as robust fallback.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx=None):
        """Update best_x and best_norm. If fx is None, it will be computed (costs a call)."""
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            if fx is None:
                fx = F_counted(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    def broyden(x_init, max_iter=30):
        """Broyden's method for 2D systems. Returns (x, success)."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-7
        
        # Initial Jacobian estimate: 3 calls
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, False
        n0 = np.linalg.norm(Fx0)
        update_best(x, Fx0)
        if n0 <= 1e-10:
            return x, True
        
        # Estimate Jacobian
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return x, False
        
        J = np.column_stack([(Fx1 - Fx0) / h, (Fx2 - Fx0) / h])
        
        # Check if Jacobian is singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-15:
            J += np.eye(2) * 1e-8
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Broyden step: solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
                step_norm = 5.0
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                dx = -dx * 0.5
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-10:
                return x_new, True
            
            # Broyden update: J <- J + (Fx_new - Fx0 - J*dx) * (dx^T) / (dx^T * dx)
            s = x_new - x
            y = Fx_new - Fx0
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
            
            x = x_new
            Fx0 = Fx_new
            n0 = n_new
        
        return x, n0 <= 1e-10
    
    def polish(x_init, max_iter=10):
        """Final high-precision polish pass using exact Newton with finite-diff Jacobian."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-8
        
        # Get current F(x)
        Fx = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x, False
        n = np.linalg.norm(Fx)
        update_best(x, Fx)
        if n <= 1e-10:
            return x, True
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Compute finite-difference Jacobian (2 calls)
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            Fx1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx2 = np.asarray(F_counted(x + e2), dtype=float)
            
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                break
            
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-15:
                J += np.eye(2) * 1e-8
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx * (1.0 / step_norm)
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-10:
                return x_new, True
            
            # Check for improvement
            if n_new > n * 1.01:
                # Step was too large, try half
                dx = dx * 0.5
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
                Fx_new = np.asarray(F_counted(x_new), dtype=float)
                if not np.all(np.isfinite(Fx_new)):
                    break
                n_new = np.linalg.norm(Fx_new)
                update_best(x_new, Fx_new)
                if n_new <= 1e-10:
                    return x_new, True
                if n_new > n:
                    break
            
            x = x_new
            Fx = Fx_new
            n = n_new
        
        return x, n <= 1e-10
    
    def nelder_mead(x_init, max_iter=50):
        """Nelder-Mead simplex method for robust global convergence."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        # Initialize simplex
        x0 = np.array(x_init, dtype=float)
        h = 0.1 * max(1.0, np.linalg.norm(x0))
        simplex = [x0.copy(), x0 + np.array([h, 0.0]), x0 + np.array([0.0, h])]
        
        # Evaluate initial simplex
        fvals = []
        for xi in simplex:
            if call_count >= MAX_CALLS or not time_ok():
                return x_init, False
            try:
                fx = np.asarray(F_counted(xi), dtype=float)
                if not np.all(np.isfinite(fx)):
                    return x_init, False
                fvals.append(np.linalg.norm(fx) ** 2)
                update_best(xi, fx)
            except:
                return x_init, False
        
        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[0] <= 1e-20:
                return simplex[0], True
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fxr = np.asarray(F_counted(xr), dtype=float)
            fxr = np.asarray(fxr, dtype=float)
            fxr_val = np.linalg.norm(fxr) ** 2 if np.all(np.isfinite(fxr)) else np.inf
            update_best(xr, fxr if np.all(np.isfinite(fxr)) else None)
            
            if fvals[0] <= fxr_val < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fxr_val
            elif fxr_val < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fxe = np.asarray(F_counted(xe), dtype=float)
                fxe = np.asarray(fxe, dtype=float)
                fxe_val = np.linalg.norm(fxe) ** 2 if np.all(np.isfinite(fxe)) else np.inf
                update_best(xe, fxe if np.all(np.isfinite(fxe)) else None)
                
                if fxe_val < fxr_val:
                    simplex[-1] = xe
                    fvals[-1] = fxe_val
                else:
                    simplex[-1] = xr
                    fvals[-1] = fxr_val
            else:
                # Contraction
                if fxr_val < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                fxc = np.asarray(F_counted(xc), dtype=float)
                fxc = np.asarray(fxc, dtype=float)
                fxc_val = np.linalg.norm(fxc) ** 2 if np.all(np.isfinite(fxc)) else np.inf
                update_best(xc, fxc if np.all(np.isfinite(fxc)) else None)
                
                if fxc_val < min(fxr_val, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fxc_val
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fx_i = np.asarray(F_counted(simplex[i]), dtype=float)
                        fx_i = np.asarray(fx_i, dtype=float)
                        fvals[i] = np.linalg.norm(fx_i) ** 2 if np.all(np.isfinite(fx_i)) else np.inf
                        update_best(simplex[i], fx_i if np.all(np.isfinite(fx_i)) else None)
        
        # Return best point
        return simplex[np.argmin(fvals)], fvals[np.argmin(fvals)] <= 1e-16
    
    # Try Broyden from x0
    x_result, success = broyden(x0)
    if success and best_norm <= 1e-8:
        return best_x
    
    # If close, try polish
    if best_norm < 0.01 and call_count < MAX_CALLS and time_ok():
        x_result, success = polish(best_x)
        if success and best_norm <= 1e-8:
            return best_x
    
    # Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = [x0.copy()]
    
    # Perturbations at various scales and directions
    for s in [0.5, 1.0, 2.0, 0.1, 5.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            candidates.append(x0 + s * np.array([dx, dy]))
    
    # Also try scaled versions
    for factor in [0.5, 2.0]:
        candidates.append(x0 * factor)
    
    # Try Broyden from each candidate
    for x_init in candidates:
        if call_count >= MAX_CALLS or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        x_result, success = broyden(x_init)
        if success and best_norm <= 1e-8:
            return best_x
        
        # If close, try polish
        if best_norm < 0.01 and call_count < MAX_CALLS and time_ok():
            x_result, success = polish(best_x)
            if success and best_norm <= 1e-8:
                return best_x
    
    # Try Nelder-Mead from best_x
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        x_result, success = nelder_mead(best_x)
        if success and best_norm <= 1e-8:
            return best_x
    
    # Try Nelder-Mead from x0
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        x_result, success = nelder_mead(x0)
        if success and best_norm <= 1e-8:
            return best_x
    
    # Final polish attempt if close
    if best_norm < 0.01 and call_count < MAX_CALLS and time_ok():
        x_result, success = polish(best_x)
        if success and best_norm <= 1e-8:
            return best_x
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
