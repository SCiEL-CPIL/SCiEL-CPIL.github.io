"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's Method.
    
    Strategy:
    1. Broyden's Method with finite-difference initial Jacobian.
       - Initial Jacobian: Central differences (5 calls).
       - Iterations: 1 call each (F(x)) + Line search overhead.
       - Jacobian update: Secant formula (0 calls).
    2. Fallback to scipy.optimize.fsolve (Levenberg-Marquardt) if Broyden fails.
    3. Random restarts if all else fails.
    
    This structure minimizes function calls by reusing Jacobian information,
    unlike standard Newton which recomputes it every step.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # Helper: Central-difference Jacobian (5 calls)
    def finite_diff_jacobian_central(x_current, h):
        Fx = compute_F(x_current)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (compute_F(x_current + e) - compute_F(x_current - e)) / (2 * h)
        return J, Fx

    # Method 1: Broyden's Method
    # Broyden's method approximates the Jacobian update, reducing calls from 3/iter (Newton) to 1/iter.
    if time_ok():
        x_broyden = x.copy()
        # Initial Jacobian
        h_init = 1e-7 * max(1.0, np.linalg.norm(x_broyden))
        try:
            J, Fx = finite_diff_jacobian_central(x_broyden, h_init)
        except Exception:
            J = None
            Fx = compute_F(x_broyden)
            
        if J is not None:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_broyden.copy()
            
            max_iters = 50
            for it in range(max_iters):
                if not time_ok():
                    break
                if norm_Fx <= 1e-10:
                    break
                
                # Solve J @ dx = -Fx
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    # If singular, try pseudo-inverse or break
                    try:
                        dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    except:
                        break
                
                # Backtracking Line Search
                alpha = 1.0
                x_trial = x_broyden + alpha * dx
                Fx_trial = compute_F(x_trial)
                norm_Fx_trial = np.linalg.norm(Fx_trial)
                
                # Check for progress
                if norm_Fx_trial < norm_Fx:
                    # Accept step
                    x_new = x_trial
                    Fx_new = Fx_trial
                    norm_Fx_new = norm_Fx_trial
                    s = x_new - x_broyden
                    y = Fx_new - Fx
                else:
                    # Reduce step
                    accepted = False
                    for _ in range(20):
                        alpha *= 0.5
                        x_trial = x_broyden + alpha * dx
                        Fx_trial = compute_F(x_trial)
                        norm_Fx_trial = np.linalg.norm(Fx_trial)
                        if norm_Fx_trial < norm_Fx:
                            x_new = x_trial
                            Fx_new = Fx_trial
                            norm_Fx_new = norm_Fx_trial
                            s = x_new - x_broyden
                            y = Fx_new - Fx
                            accepted = True
                            break
                    if not accepted:
                        # No progress, break
                        break
                
                # Update best
                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()
                
                # Broyden Update: J_new = J + (y - J @ s) @ s.T / (s.T @ s)
                if np.linalg.norm(s) > 1e-12:
                    J = J + np.outer(y - J @ s, s) / np.dot(s, s)
                else:
                    # Stagnation in position, recompute Jacobian
                    h_re = 1e-7 * max(1.0, np.linalg.norm(x_new))
                    J, Fx_new = finite_diff_jacobian_central(x_new, h_re)
                
                x_broyden = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new

    # Check if Broyden converged
    if best_norm <= 1e-8:
        return best_x

    # Method 2: SciPy fsolve fallback
    # fsolve is highly robust and uses C-optimized routines.
    # We limit maxfev to prevent excessive calls if it struggles.
    if time_ok():
        try:
            # Use x0 as start, or best_x if it's better
            start_point = best_x
            # fsolve returns x and info. info > 0 means success.
            # maxfev limits the number of function evaluations.
            # We have used ~50 calls so far. Budget is 500. Allow 400 more.
            sol, info, ier, msg = so.fsolve(
                lambda x: np.asarray(F(x), dtype=float), 
                start_point, 
                full_output=True, 
                maxfev=400, 
                xtol=1e-10
            )
            if ier == 1:
                norm_sol = np.linalg.norm(compute_F(sol))
                if norm_sol < best_norm:
                    best_norm = norm_sol
                    best_x = sol.copy()
                if best_norm <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Method 3: Random restarts with simple Newton (if still not converged)
    if best_norm > 1e-8 and time_ok():
        for attempt in range(10):
            if not time_ok():
                break
            # Perturb from best_x
            x_try = best_x + np.random.randn(2) * 0.5
            x_cur = x_try.copy()
            
            # Simple Newton with forward diff (3 calls/iter)
            for it in range(30):
                if not time_ok():
                    break
                Fx = compute_F(x_cur)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_cur.copy()
                if norm_Fx <= 1e-10:
                    break
                
                h = 1e-7 * max(1.0, np.linalg.norm(x_cur))
                # Forward diff Jacobian (3 calls: Fx already have, need Fx+e1, Fx+e2)
                Fx0 = Fx
                e1 = np.array([h, 0.0])
                e2 = np.array([0.0, h])
                J = np.zeros((2, 2))
                J[:, 0] = (compute_F(x_cur + e1) - Fx0) / h
                J[:, 1] = (compute_F(x_cur + e2) - Fx0) / h
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                
                # Backtrack
                alpha = 1.0
                x_new = x_cur + alpha * dx
                Fx_new = compute_F(x_new)
                if np.linalg.norm(Fx_new) < norm_Fx:
                    x_cur = x_new
                else:
                    # Try smaller step
                    alpha = 0.5
                    x_new = x_cur + alpha * dx
                    Fx_new = compute_F(x_new)
                    if np.linalg.norm(Fx_new) < norm_Fx:
                        x_cur = x_new
                    else:
                        break

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END
