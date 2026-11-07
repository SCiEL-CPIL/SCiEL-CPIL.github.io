"""Root finder for 3 fixed 2D systems using Nelder-Mead global search + Newton refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 148.0  # Safety margin from 150s limit
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 495
    
    def safe_eval(x):
        if call_counter[0] >= max_calls:
            return None, float('inf')
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, float('inf')
            call_counter[0] += 1
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, float('inf')
    
    def merit(x):
        """Scalar merit function: ||F(x)||^2"""
        Fx, norm_F = safe_eval(x)
        if Fx is None:
            return float('inf')
        return norm_F ** 2
    
    def check_time():
        return (time.time() - start_time) > time_limit
    
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    # Check if x0 is already a root
    Fx, norm_F = safe_eval(x0_arr)
    if Fx is not None:
        best_norm = norm_F
        if norm_F <= 1e-8:
            return best_x
    
    # === Phase 1: Nelder-Mead global search on ||F(x)||^2 ===
    # Initialize simplex around x0 with adaptive scale
    def nelder_mead(start_x, max_iters=300, initial_scale=1.0):
        nonlocal best_x, best_norm, call_counter
        
        # Build initial simplex
        if initial_scale < 1e-6:
            initial_scale = 1.0
        
        # Create simplex: center + 2 vertices
        v0 = start_x.copy()
        # Use small perturbations
        v1 = v0.copy()
        v1[0] += initial_scale
        v2 = v0.copy()
        v2[1] += initial_scale
        
        simplex = [v0.copy(), v1.copy(), v2.copy()]
        fvals = []
        
        for v in simplex:
            f = merit(v)
            fvals.append(f)
            if f < best_norm ** 2:
                best_norm = np.sqrt(f)
                best_x = v.copy()
        
        if best_norm <= 1e-8:
            return True
        
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        for it in range(max_iters):
            if check_time() or call_counter[0] >= max_calls:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if fvals[-1] - fvals[0] < 1e-14:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = merit(x_r)
            
            if fvals[0] <= f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            elif f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = merit(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = merit(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = merit(simplex[i])
            
            # Update best
            min_f = min(fvals)
            if min_f < best_norm ** 2:
                best_norm = np.sqrt(min_f)
                best_x = simplex[np.argmin(fvals)].copy()
            
            if best_norm <= 1e-8:
                return True
        
        return best_norm <= 1e-8
    
    # Run Nelder-Mead from x0 with several scales
    scales = [1.0, 0.1, 10.0, 0.01, 100.0]
    for scale in scales:
        if check_time() or call_counter[0] >= max_calls:
            break
        if nelder_mead(x0_arr, max_iters=200, initial_scale=scale):
            break
    
    # Also try Nelder-Mead from best_x found so far
    if not (best_norm <= 1e-8):
        if not check_time() and call_counter[0] < max_calls:
            nelder_mead(best_x, max_iters=150, initial_scale=0.1)
    
    # === Phase 2: Newton/Broyden refinement from best point ===
    if best_norm > 1e-8 and not check_time() and call_counter[0] < max_calls:
        # Use Broyden's method starting from best_x
        x = best_x.copy()
        Fx, norm_F = safe_eval(x)
        
        if Fx is not None and np.all(np.isfinite(Fx)):
            # Initialize Jacobian approximation using finite differences
            # This costs 2 extra calls but gives a good initial J
            eps = 1e-7
            J = np.zeros((2, 2))
            
            for j in range(2):
                x_pert = x.copy()
                x_pert[j] += eps
                Fx_pert, _ = safe_eval(x_pert)
                if Fx_pert is not None:
                    J[:, j] = (Fx_pert - Fx) / eps
                else:
                    J[:, j] = 1.0  # fallback
            
            # Broyden iterations
            for _ in range(50):
                if check_time() or call_counter[0] >= max_calls:
                    break
                
                # Solve J * s = -Fx
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    try:
                        s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    except Exception:
                        break
                
                if not np.all(np.isfinite(s)):
                    break
                
                # Damped Newton step with backtracking
                alpha = 1.0
                x_new = None
                Fx_new = None
                norm_new = float('inf')
                
                for _ in range(20):
                    x_trial = x + alpha * s
                    Fx_trial, norm_trial = safe_eval(x_trial)
                    
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                    
                    if norm_trial < (1 - 1e-4 * alpha) * norm_F:
                        x_new = x_trial
                        Fx_new = Fx_trial
                        norm_new = norm_trial
                        break
                    else:
                        alpha *= 0.5
                
                if x_new is None:
                    break
                
                # Update Broyden
                delta_x = x_new - x
                delta_F = Fx_new - Fx
                denom = np.dot(delta_x, delta_x)
                
                if denom > 1e-14:
                    J = J + np.outer(delta_F - J @ delta_x, delta_x) / denom
                
                x = x_new
                Fx = Fx_new
                norm_F = norm_new
                
                if norm_F < best_norm:
                    best_norm = norm_F
                    best_x = x.copy()
                
                if norm_F <= 1e-8:
                    break
    
    # Final safety: return best_x
    if not np.all(np.isfinite(best_x)):
        best_x = x0_arr.copy()
    
    return best_x
# EVOLVE-BLOCK-END
