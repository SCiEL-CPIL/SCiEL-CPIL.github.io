"""Root finder for 3 fixed 2D systems using multi-start Newton with adaptive safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _newton_step(F, x, Fx, tol=1e-12, max_iter=40, h=1e-7):
    """Newton's method with finite-difference Jacobian. Returns (x, Fx, calls_used)."""
    calls = 1  # Fx already computed
    x = np.array(x, dtype=float)
    Fx = np.array(Fx, dtype=float)
    
    for _ in range(max_iter):
        if np.linalg.norm(Fx) <= tol:
            return x, Fx, calls
        
        # Compute finite-difference Jacobian
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = F(x + e)
            calls += 1
            J[:, j] = (Fx_plus - Fx) / h
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, try a smaller step
            try:
                dx = np.linalg.lstsq(J, Fx, rcond=None)[0]
            except Exception:
                break
        
        # Damped Newton step
        step_norm = np.linalg.norm(dx)
        if step_norm == 0:
            break
        
        # Try full step first
        x_new = x - dx
        Fx_new = F(x_new)
        calls += 1
        
        if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
            x = x_new
            Fx = Fx_new
        else:
            # Backtracking
            alpha = 0.5
            for _ in range(10):
                x_new = x - alpha * dx
                Fx_new = F(x_new)
                calls += 1
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    x = x_new
                    Fx = Fx_new
                    break
                alpha *= 0.5
            else:
                # No improvement, break
                break
    
    return x, Fx, calls


def find_root_system(F, x0):
    """Find x in R^2 such that ||F(x)|| <= 1e-8."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    x0 = np.array(x0, dtype=float)
    
    # Track best result found
    best_x = x0.copy()
    best_Fx = F(x0)
    best_norm = np.linalg.norm(best_Fx)
    total_calls = 1
    
    # If already good enough
    if best_norm <= 1e-8:
        return best_x
    
    # Generate multiple starting points: x0 plus angular sampling at various radii
    starts = [x0.copy()]
    
    # Radii to explore: small to large
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_angles = 8
    
    for r in radii:
        for k in range(n_angles):
            angle = 2.0 * np.pi * k / n_angles
            x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
            starts.append(x_start)
    
    # Sort starts by distance from x0 (try closest first)
    starts.sort(key=lambda x: np.linalg.norm(x - x0))
    
    for x_start in starts:
        if time.time() >= deadline:
            break
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        # Try Newton from this start
        try:
            Fx_start = F(x_start)
            total_calls += 1
        except Exception:
            continue
        
        if not np.all(np.isfinite(Fx_start)):
            continue
        
        norm_start = np.linalg.norm(Fx_start)
        
        # Update best if this start is better
        if norm_start < best_norm:
            best_x = x_start.copy()
            best_Fx = Fx_start.copy()
            best_norm = norm_start
        
        if best_norm <= 1e-8:
            return best_x
        
        # Run Newton
        x_result, Fx_result, calls_used = _newton_step(F, x_start, Fx_start)
        total_calls += calls_used - 1  # Fx_start already counted
        
        norm_result = np.linalg.norm(Fx_result)
        
        if norm_result < best_norm:
            best_x = x_result.copy()
            best_Fx = Fx_result.copy()
            best_norm = norm_result
        
        if best_norm <= 1e-8:
            return best_x
    
    # If we haven't found a good root yet, try a few more strategies
    if time.time() < deadline and best_norm > 1e-8:
        # Try larger radii with fewer angles
        for r in [200.0, 500.0, 1000.0]:
            if time.time() >= deadline:
                break
            for k in range(4):
                angle = 2.0 * np.pi * k / 4 + 0.785  # offset
                x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
                
                if not np.all(np.isfinite(x_start)):
                    continue
                
                try:
                    Fx_start = F(x_start)
                    total_calls += 1
                except Exception:
                    continue
                
                if not np.all(np.isfinite(Fx_start)):
                    continue
                
                norm_start = np.linalg.norm(Fx_start)
                
                if norm_start < best_norm:
                    best_x = x_start.copy()
                    best_Fx = Fx_start.copy()
                    best_norm = norm_start
                
                if best_norm <= 1e-8:
                    return best_x
                
                x_result, Fx_result, calls_used = _newton_step(F, x_start, Fx_start)
                total_calls += calls_used - 1
                
                norm_result = np.linalg.norm(Fx_result)
                
                if norm_result < best_norm:
                    best_x = x_result.copy()
                    best_Fx = Fx_result.copy()
                    best_norm = norm_result
                
                if best_norm <= 1e-8:
                    return best_x
    
    return best_x
# EVOLVE-BLOCK-END
