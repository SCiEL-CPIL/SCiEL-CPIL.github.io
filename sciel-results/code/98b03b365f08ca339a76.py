"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def find_root_system(F, x0):
    start_time = time.time()
    # Evaluator kills at 150s. We must return before then. 
    # Use 145s as a hard deadline to be safe.
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def is_good(x, Fx=None):
        if not np.all(np.isfinite(x)):
            return False
        try:
            if Fx is None:
                Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False

    def initial_jac(F, x, Fx, eps=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                Fx_e = F(x + e)
                if not np.all(np.isfinite(Fx_e)):
                    return None
                J[:, j] = (Fx_e - Fx) / eps
            except Exception:
                return None
        if not np.all(np.isfinite(J)):
            return None
        return J

    def broyden_step(F, x, J, Fx, max_iter=100, tol=1e-12):
        """
        Broyden's Good Method for root finding.
        Updates J using rank-1 update.
        Returns (x_best, Fx_best, calls_made)
        """
        x_curr = x.copy()
        J_curr = J.copy()
        Fx_curr = Fx.copy()
        calls = 0 # We already have Fx for x0, so 0 extra calls for this state
        
        best_x = x_curr.copy()
        best_Fx = Fx_curr.copy()
        best_norm = norm(Fx_curr)
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
            
            current_norm = norm(Fx_curr)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x_curr.copy()
                best_Fx = Fx_curr.copy()
            
            if current_norm <= tol:
                return best_x, best_Fx, calls

            # Solve J dx = -Fx
            try:
                # Check condition number roughly to avoid instability
                det = np.linalg.det(J_curr)
                if abs(det) < 1e-12:
                    # Singular or near singular, perturb
                    J_curr += 1e-6 * np.eye(2)
                    det = np.linalg.det(J_curr)
                    if abs(det) < 1e-12:
                        break
                dx = np.linalg.solve(J_curr, -Fx_curr)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Damping / Line Search
            # Try full step first
            step_accepted = False
            lam = 1.0
            
            # Simple backtracking
            for _ in range(10):
                x_new = x_curr + lam * dx
                try:
                    Fx_new = F(x_new)
                    calls += 1
                except Exception:
                    lam *= 0.5
                    continue
                
                if not np.all(np.isfinite(Fx_new)):
                    lam *= 0.5
                    continue
                
                new_norm = norm(Fx_new)
                
                # Accept if it reduces norm
                if new_norm < current_norm:
                    # Broyden Good Update: J_new = J + (Fx_new - Fx - J*dx) * dx^T / (dx^T dx)
                    # Vector form: J += (delta_F - J @ delta_x) @ (delta_x^T) / (delta_x @ delta_x)
                    delta_F = Fx_new - Fx_curr
                    J_dx = J_curr @ dx
                    denom = np.dot(dx, dx)
                    if denom > 1e-15:
                        J_curr += np.outer((delta_F - J_dx), dx) / denom
                    
                    x_curr = x_new
                    Fx_curr = Fx_new
                    step_accepted = True
                    break
                else:
                    lam *= 0.5
            
            if not step_accepted:
                break
                
            if norm(dx) < 1e-12:
                break

        return best_x, best_Fx, calls

    # --- Main Logic ---
    
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
    
    # Phase 1: Try Broyden from x0
    try:
        Fx0 = F(x0)
        if not np.all(np.isfinite(Fx0)):
            Fx0 = None
    except Exception:
        Fx0 = None
        
    if Fx0 is not None and norm(Fx0) <= 1e-8:
        return x0

    x_best = x0.copy()
    Fx_best = Fx0.copy() if Fx0 is not None else np.array([np.inf, np.inf])
    best_norm = norm(Fx_best) if Fx0 is not None else np.inf
    
    if Fx0 is not None:
        J0 = initial_jac(F, x0, Fx0)
        if J0 is not None:
            x_b, Fx_b, _ = broyden_step(F, x0, J0, Fx0)
            if norm(Fx_b) < best_norm:
                x_best = x_b
                Fx_best = Fx_b
                best_norm = norm(Fx_b)
                if best_norm <= 1e-8:
                    return x_best

    # Phase 2: Global Search using Nelder-Mead on ||F(x)||^2
    # This is robust for finding basins of attraction.
    # We define a wrapper function for scipy.
    
    def objective(x_vec):
        try:
            Fx = F(x_vec)
            if not np.all(np.isfinite(Fx)):
                return 1e10
            return np.dot(Fx, Fx) # Minimize squared norm
        except Exception:
            return 1e10

    # Generate initial simplex for Nelder-Mead
    # Start around x0 and best found
    points = [x0, x_best]
    
    # Add some diverse points
    scale = max(1.0, norm(x0))
    rng = np.random.default_rng(42)
    
    # Add random points in a sphere around 0 and x0
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        u = rng.normal(size=2)
        u /= norm(u)
        points.append(x0 + r * u)
        points.append(r * u)
        
    # Also try zero
    points.append(np.array([0.0, 0.0]))
    
    # Deduplicate and filter finite
    unique_points = []
    for p in points:
        if not np.all(np.isfinite(p)):
            continue
        # Simple check if close to existing
        is_dup = False
        for up in unique_points:
            if norm(p - up) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)
            
    if not unique_points:
        unique_points = [np.array([0.0, 0.0])]
        
    # Use scipy.optimize.minimize with Nelder-Mead
    # maxiter is set high, but we check time inside objective? No, objective can't check time easily without global state.
    # We will run minimize with a reasonable maxiter and rely on the deadline check in the main loop if we implement a custom callback or just run it and check time after.
    # Since minimize runs in C, it might ignore time_left. 
    # Strategy: Run minimize with a limited maxiter (e.g., 100) and multiple restarts from the best points found.
    
    from scipy.optimize import minimize
    
    best_x_nm = x_best
    best_f_nm = best_norm
    
    # We will try NM from the best few points
    start_points = unique_points[:5]
    
    for sp in start_points:
        if time_left() < 5.0:
            break
            
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.success or res.fun < best_f_nm:
                if np.all(np.isfinite(res.x)):
                    Fx_res = F(res.x)
                    if np.all(np.isfinite(Fx_res)):
                        n_res = norm(Fx_res)
                        if n_res < best_f_nm:
                            best_f_nm = n_res
                            best_x_nm = res.x
                            if best_f_nm <= 1e-8:
                                return best_x_nm
        except Exception:
            pass
            
    # Phase 3: Polish with Broyden from the best NM result
    if best_f_nm < best_norm:
        x_best = best_x_nm
        Fx_best = F(best_x_nm)
        best_norm = best_f_nm
        
    # If we still haven't converged, try Broyden from x_best
    if best_norm > 1e-8 and time_left() > 2.0:
        try:
            Fx_start = F(x_best)
            if np.all(np.isfinite(Fx_start)):
                J_start = initial_jac(F, x_best, Fx_start)
                if J_start is not None:
                    x_pol, Fx_pol, _ = broyden_step(F, x_best, J_start, Fx_start)
                    n_pol = norm(Fx_pol)
                    if n_pol < best_norm:
                        x_best = x_pol
                        Fx_best = Fx_pol
                        best_norm = n_pol
                        if best_norm <= 1e-8:
                            return x_best
        except Exception:
            pass

    # Final check
    if is_good(x_best, Fx_best):
        return x_best

    # If still not good, return the best we have
    return x_best

# EVOLVE-BLOCK-END
