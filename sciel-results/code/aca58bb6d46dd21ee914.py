"""Root finder for 3 fixed 2D systems with adaptive damping and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (2,):
        x0 = x0.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def _time_left():
        return deadline - time.time()
    
    def _is_good(x, Fx):
        if Fx is None:
            return False
        if not np.all(np.isfinite(x)) or x.shape != (2,):
            return False
        if not np.all(np.isfinite(Fx)):
            return False
        return np.linalg.norm(Fx) <= 1e-8
    
    # First, try to get a good result from x0 with damped Newton
    x = x0.copy()
    best_x = x.copy()
    best_Fx = None
    try:
        best_Fx = F(x)
    except:
        best_Fx = None
    
    if best_Fx is not None and _is_good(x, best_Fx):
        return x
    
    # Damped Newton with adaptive step size
    def newton_damped(x, max_iter=40):
        x = x.copy()
        Fx = None
        try:
            Fx = F(x)
        except:
            return None, None
        
        if Fx is None or not np.all(np.isfinite(Fx)):
            return None, None
        
        if _is_good(x, Fx):
            return x, Fx
        
        for it in range(max_iter):
            if _time_left() < 1.0:
                break
            
            # Compute Jacobian via finite differences
            J = np.zeros((2, 2))
            h = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Fx_plus = F(x + e)
                    Fx_minus = F(x - e)
                    if Fx_plus is not None and Fx_minus is not None and np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus)):
                        J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                    else:
                        J[:, j] = (Fx_plus - Fx) / h if Fx_plus is not None and np.all(np.isfinite(Fx_plus)) else 0
                except:
                    J[:, j] = 0
            
            # Check if Jacobian is valid
            if not np.all(np.isfinite(J)):
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Adaptive damping
            step = 1.0
            for _ in range(20):
                x_new = x + step * dx
                try:
                    Fx_new = F(x_new)
                except:
                    step *= 0.5
                    continue
                
                if Fx_new is None or not np.all(np.isfinite(Fx_new)):
                    step *= 0.5
                    continue
                
                norm_new = np.linalg.norm(Fx_new)
                norm_old = np.linalg.norm(Fx)
                
                if norm_new < norm_old * 0.9 or norm_new <= 1e-8:
                    x = x_new
                    Fx = Fx_new
                    break
                else:
                    step *= 0.5
            
            if _is_good(x, Fx):
                return x, Fx
            
            if np.linalg.norm(dx) * step < 1e-14:
                break
        
        return x, Fx
    
    # Try Newton from x0
    result, Fx_result = newton_damped(x0)
    if result is not None and Fx_result is not None:
        if _is_good(result, Fx_result):
            return result
        if best_Fx is None or np.linalg.norm(Fx_result) < np.linalg.norm(best_Fx):
            best_x = result
            best_Fx = Fx_result
    
    # If we have a good enough result, return it
    if best_Fx is not None and np.linalg.norm(best_Fx) <= 1e-8:
        return best_x
    
    # Try multiple restarts if we have time
    if _time_left() > 5.0:
        # Generate diverse starting points
        starts = [x0.copy(), best_x.copy()]
        
        # Add perturbations around best_x
        for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
            for dx in [0.1, -0.1, 0.5, -0.5]:
                s = best_x.copy()
                s[0] += dx * scale
                s[1] += dx * scale
                starts.append(s)
        
        # Add some random-looking starts based on x0
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]:
            r = np.linalg.norm(x0)
            if r < 1e-10:
                r = 1.0
            s = np.array([r * np.cos(angle), r * np.sin(angle)])
            starts.append(s)
        
        for s in starts:
            if _time_left() < 2.0:
                break
            
            if not np.all(np.isfinite(s)):
                continue
            
            result, Fx_result = newton_damped(s, max_iter=30)
            if result is not None and Fx_result is not None:
                if _is_good(result, Fx_result):
                    return result
                if best_Fx is None or np.linalg.norm(Fx_result) < np.linalg.norm(best_Fx):
                    best_x = result
                    best_Fx = Fx_result
        
        # If we still have time, try more aggressive restarts
        if _time_left() > 5.0 and best_Fx is not None and np.linalg.norm(best_Fx) > 1e-8:
            for trial in range(50):
                if _time_left() < 1.0:
                    break
                
                # Random direction and magnitude
                angle = trial * 2 * np.pi / 50
                magnitudes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
                for mag in magnitudes:
                    if _time_left() < 0.5:
                        break
                    
                    s = best_x + mag * np.array([np.cos(angle), np.sin(angle)])
                    if not np.all(np.isfinite(s)):
                        continue
                    
                    result, Fx_result = newton_damped(s, max_iter=25)
                    if result is not None and Fx_result is not None:
                        if _is_good(result, Fx_result):
                            return result
                        if best_Fx is None or np.linalg.norm(Fx_result) < np.linalg.norm(best_Fx):
                            best_x = result
                            best_Fx = Fx_result
    
    # Return the best result we found
    if best_Fx is not None and np.all(np.isfinite(best_x)):
        return best_x
    
    # Last resort: return x0 or best_x
    if np.all(np.isfinite(best_x)):
        return best_x
    
    return x0.copy()
# EVOLVE-BLOCK-END
