import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method (1 F call/iter) for rapid convergence.
    2. Initial Jacobian via forward differences (3 calls total).
    3. LM solver with complex-step Jacobian (2 calls/iter) when Broyden stalls.
    4. Final polish with quadratic model (3 calls/iter) or Newton for high precision.
    5. Strict call counting and time management.
    6. Global search with Sobol-like sampling to identify basins of attraction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 450  # Keep well under 500 to avoid penalty
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            fx = np.asarray(fx, dtype=float)
            if x.shape != (2,) or fx.shape != (2,):
                return
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    # Helper: Complex-step Jacobian (2 calls)
    def jac_complex(x):
        h = 1e-30
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        
        Fx_c1 = F_counted(x + 1j * h * e1)
        Fx_c2 = F_counted(x + 1j * h * e2)
        
        # Complex-step derivative: dF/dx_j = Im(F(x + i*h*e_j)) / h
        col1 = np.imag(Fx_c1) / h
        col2 = np.imag(Fx_c2) / h
        
        if not (np.all(np.isfinite(col1)) and np.all(np.isfinite(col2))):
            return None
        J = np.column_stack([col1, col2])
        return J

    # Helper: Central Difference Jacobian (4 calls)
    def jac_central(x):
        h = 1e-7 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_p1 = F_counted(x + e1)
        Fx_m1 = F_counted(x - e1)
        Fx_p2 = F_counted(x + e2)
        Fx_m2 = F_counted(x - e2)
        
        if not (np.all(np.isfinite(Fx_p1)) and np.all(np.isfinite(Fx_m1)) and 
                np.all(np.isfinite(Fx_p2)) and np.all(np.isfinite(Fx_m2))):
            return None
        J = np.column_stack([(Fx_p1 - Fx_m1) / (2*h), (Fx_p2 - Fx_m2) / (2*h)])
        return J

    # Helper: Forward Difference Jacobian (3 calls including current F)
    def jac_forward(x, Fx=None):
        h = 1e-6 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        if Fx is None:
            Fx = F_counted(x)
        
        Fx_p1 = F_counted(x + e1)
        Fx_p2 = F_counted(x + e2)
        
        if not (np.all(np.isfinite(Fx_p1)) and np.all(np.isfinite(Fx_p2)) and np.all(np.isfinite(Fx))):
            return None, Fx
        J = np.column_stack([(Fx_p1 - Fx) / h, (Fx_p2 - Fx) / h])
        return J, Fx

    # Core Broyden Solver
    def solve_broyden(x_init, max_iter=50):
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return False
        
        x = np.array(x_init, dtype=float)
        
        # 1. Initial F and Jacobian
        Fx = F_counted(x)
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= 1e-8:
            return True
        
        J, Fx = jac_forward(x, Fx)
        if J is None:
            return False
            
        # Ensure J is not singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-10:
            J += np.eye(2) * 1e-8
            
        prev_norm = np.linalg.norm(Fx)
        stagnation_count = 0
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 5 or not time_ok():
                break
            
            # 2. Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # 3. Damping if step is too large
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            elif step_norm < 1e-14:
                break
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                # Try opposite direction if non-finite
                x_new = x - dx
                if not np.all(np.isfinite(x_new)):
                    break
            
            Fx_new = F_counted(x_new)
            if not np.all(np.isfinite(Fx_new)):
                break
                
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-8:
                return True
            
            # Stagnation check
            if n_new > prev_norm * 0.9:
                stagnation_count += 1
                if stagnation_count >= 3:
                    break
            else:
                stagnation_count = 0
            prev_norm = n_new
            
            # 4. Broyden Update (Good Method)
            s = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s, s)
            if denom > 1e-20:
                J = J + np.outer(y - J @ s, s) / denom
            
            x = x_new
            Fx = Fx_new
            
        return best_norm <= 1e-8

    # Levenberg-Marquardt solver with complex-step Jacobian
    def solve_lm(x_init, max_iter=30):
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS - 10 or not time_ok():
            return False
        
        x = np.array(x_init, dtype=float)
        Fx = F_counted(x)
        update_best(x, Fx)
        n_curr = np.linalg.norm(Fx)
        
        if n_curr <= 1e-8:
            return True
        
        # LM damping parameter - start with moderate value
        lam = 1e-2
        lam_min = 1e-12
        lam_max = 1e4
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 3 or not time_ok():
                break
            
            # Complex-step Jacobian (2 calls)
            J = jac_complex(x)
            if J is None:
                break
            
            # LM step: (J^T J + lam * diag(J^T J)) dx = -J^T F
            JtJ = J.T @ J
            JtF = J.T @ Fx
            diag_JtJ = np.diag(JtJ).copy()
            diag_JtJ[diag_JtJ < 1e-14] = 1e-14
            
            A = JtJ + lam * np.diag(diag_JtJ)
            b = -JtF
            
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                x_new = x - dx
                if not np.all(np.isfinite(x_new)):
                    break
            
            Fx_new = F_counted(x_new)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-8:
                return True
            
            # Adjust lambda based on improvement ratio
            if n_new < n_curr:
                # Good step - reduce lambda to take bigger Newton steps
                x = x_new
                Fx = Fx_new
                n_curr = n_new
                lam = max(lam * 0.2, lam_min)
            else:
                # Bad step - increase lambda for more damping
                lam = min(lam * 5.0, lam_max)
                if lam > 1e3:
                    break
        
        return best_norm <= 1e-8

    # Core Polisher using quadratic model or Newton
    def polish(x_init, max_iter=20):
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS - 10 or not time_ok():
            return False
        
        x = np.array(x_init, dtype=float)
        Fx = F_counted(x)
        update_best(x, Fx)
        n = np.linalg.norm(Fx)
        
        if n <= 1e-8:
            return True
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 2 or not time_ok():
                break
            
            if n > 1e-2:
                # Too far for polish, break back to main solver
                break
            
            # Choose method based on how close we are
            if n < 1e-4:
                # Very close: use complex-step Jacobian Newton (2 calls)
                J = jac_complex(x)
                if J is None:
                    J = jac_central(x)
                    if J is None:
                        break
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    try:
                        dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    except:
                        break
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
                Fx_new = F_counted(x_new)
                update_best(x_new, Fx_new)
                n_new = np.linalg.norm(Fx_new)
                if n_new < n:
                    x = x_new
                    Fx = Fx_new
                    n = n_new
                else:
                    break
            else:
                # Use quadratic model step (3 calls per iteration)
                dx, Fx, _, _ = quad_model_step(x, Fx)
                if dx is None:
                    # Fall back to Newton
                    J = jac_complex(x)
                    if J is None:
                        J = jac_central(x)
                        if J is None:
                            break
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        try:
                            dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                        except:
                            break
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
                Fx_new = F_counted(x_new)
                update_best(x_new, Fx_new)
                n_new = np.linalg.norm(Fx_new)
                if n_new < n:
                    x = x_new
                    Fx = Fx_new
                    n = n_new
                else:
                    # Try damped steps
                    found = False
                    for damp in [0.5, 0.25, 0.1]:
                        x_damp = x + damp * dx
                        if not np.all(np.isfinite(x_damp)):
                            continue
                        Fx_damp = F_counted(x_damp)
                        update_best(x_damp, Fx_damp)
                        n_damp = np.linalg.norm(Fx_damp)
                        if n_damp < n:
                            x = x_damp
                            Fx = Fx_damp
                            n = n_damp
                            found = True
                            break
                    if not found:
                        break
                    
        return best_norm <= 1e-8

    # Helper: Quadratic model step using 3 evaluations (x, x+e1, x+e2)
    def quad_model_step(x, Fx, h=None):
        """
        Estimate a quadratic model of g(x) = ||F(x)||^2 using 3 evaluations.
        Returns the step dx to take, or None if failed.
        """
        if h is None:
            h = 1e-5 * max(1.0, np.linalg.norm(x))
        
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        g0 = np.dot(Fx, Fx)
        Fx1 = F_counted(x + e1)
        Fx2 = F_counted(x + e2)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return None, Fx, Fx1, Fx2
        
        g1 = np.dot(Fx1, Fx1)
        g2 = np.dot(Fx2, Fx2)
        
        # Estimate gradient via forward differences
        grad1 = (g1 - g0) / h
        grad2 = (g2 - g0) / h
        
        grad = np.array([grad1, grad2])
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            return None, Fx, Fx1, Fx2
        
        # Steepest descent direction
        d = -grad / grad_norm
        
        # Adaptive line search: try several step sizes
        best_dx = None
        best_g = g0
        
        # Try decreasing step sizes
        for alpha in [h, h*0.5, h*0.25, h*0.1, h*0.05]:
            dx = alpha * d
            x_trial = x + dx
            if not np.all(np.isfinite(x_trial)):
                continue
            Fx_trial = F_counted(x_trial)
            if not np.all(np.isfinite(Fx_trial)):
                continue
            g_trial = np.dot(Fx_trial, Fx_trial)
            if g_trial < best_g:
                best_g = g_trial
                best_dx = dx
        
        if best_dx is None:
            # No improvement found, return None
            return None, Fx, Fx1, Fx2
        
        return best_dx, Fx, Fx1, Fx2

    # Global search using Sobol-like sampling in logarithmic radius
    def global_search(x0, max_evals=30):
        """
        Sample points in a logarithmic radius around origin and x0.
        Return the point with the lowest residual norm.
        """
        if call_count >= MAX_CALLS - 10 or not time_ok():
            return None
        
        scale = max(1.0, np.linalg.norm(x0))
        
        # Generate Sobol-like points in 2D
        # Use a simple low-discrepancy sequence
        n_points = min(max_evals, MAX_CALLS - call_count - 10)
        
        # Logarithmic radii: from 0.01*scale to 100*scale
        radii = np.logspace(-2, 2, 8) * scale
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        
        candidates = []
        for r in radii:
            for a in angles:
                x = np.array([r * np.cos(a), r * np.sin(a)])
                candidates.append(x)
        
        # Also add points around x0
        for r in radii:
            for a in angles:
                x = x0 + np.array([r * np.cos(a), r * np.sin(a)])
                candidates.append(x)
        
        # Add origin and axis points
        candidates.append(np.array([0.0, 0.0]))
        candidates.append(np.array([scale, 0.0]))
        candidates.append(np.array([-scale, 0.0]))
        candidates.append(np.array([0.0, scale]))
        candidates.append(np.array([0.0, -scale]))
        
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        best_local_x = None
        best_local_norm = np.inf
        
        for x in candidates:
            if call_count >= MAX_CALLS - 10 or not time_ok():
                break
            
            if not np.all(np.isfinite(x)):
                continue
            
            Fx = F_counted(x)
            if not np.all(np.isfinite(Fx)):
                continue
            
            n = np.linalg.norm(Fx)
            update_best(x, Fx)
            
            if n < best_local_norm:
                best_local_norm = n
                best_local_x = x.copy()
            
            if n <= 1e-8:
                return x
        
        return best_local_x

    # --- Execution Plan ---
    
    # 1. Try Broyden from x0
    if solve_broyden(x0):
        return best_x
    
    # 2. If Broyden didn't converge but we're in a reasonable range, try LM
    if best_norm < 1.0 and call_count < MAX_CALLS - 20 and time_ok():
        if solve_lm(best_x):
            return best_x
        
    # 3. If close, try to polish
    if best_norm < 1e-2 and call_count < MAX_CALLS - 10 and time_ok():
        if polish(best_x):
            return best_x
            
    # 4. Global search to find a better starting point
    if call_count < MAX_CALLS - 30 and time_ok():
        global_x = global_search(x0, max_evals=20)
        if global_x is not None:
            # Try Broyden from the global search result
            if solve_broyden(global_x):
                return best_x
            
            # Try LM from this point
            if best_norm < 1.0 and call_count < MAX_CALLS - 15 and time_ok():
                if solve_lm(best_x):
                    return best_x
                
            # Try polish
            if best_norm < 1e-2 and call_count < MAX_CALLS - 10 and time_ok():
                if polish(best_x):
                    return best_x

    # 5. Restart with perturbations if not converged
    # Generate smart restart points with varied scales
    scale = max(1.0, np.linalg.norm(x0))
    restarts = [
        x0 + np.array([scale, 0.0]),
        x0 + np.array([0.0, scale]),
        x0 + np.array([scale, scale]) * 0.5,
        x0 * 0.5,
        np.array([0.0, 0.0]),
        x0 + np.array([scale, -scale]) * 0.5,
        x0 - np.array([scale, 0.0]),
        x0 - np.array([0.0, scale]),
        x0 + np.array([-scale, scale]) * 0.5,
        x0 * 2.0,
        x0 + np.array([-scale, -scale]) * 0.5,
        x0 * 0.25,
        x0 * 4.0,
    ]
    
    for r in restarts:
        if call_count >= MAX_CALLS - 25 or not time_ok():
            break
        if not np.all(np.isfinite(r)):
            continue
        
        if solve_broyden(r):
            return best_x
        
        # Try LM from this restart if Broyden didn't work
        if best_norm < 1.0 and call_count < MAX_CALLS - 15 and time_ok():
            if solve_lm(best_x):
                return best_x
            
        if best_norm < 1e-2 and call_count < MAX_CALLS - 10 and time_ok():
            if polish(best_x):
                return best_x
                
    # 6. Final desperate polish if we are close
    if best_norm < 1e-1 and call_count < MAX_CALLS - 5 and time_ok():
        if polish(best_x):
            return best_x

    # Return best found
    return best_x if np.all(np.isfinite(best_x)) else x0
