"""Root finder for 3 fixed 2D systems using trust-region dogleg with Broyden Hessian approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: Trust-region dogleg with Broyden Hessian approximation.
    Fallback: Multi-start Broyden, homotopy continuation, SciPy.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def time_up():
        return time.time() > deadline

    # --- Phase 1: Trust-region dogleg with Broyden Hessian ---
    result = _trust_region_dogleg(F, x, start_time, deadline)
    if result is not None:
        try:
            Fx_res = F(result)
            if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) <= 1e-8:
                return result
        except Exception:
            pass
    
    # --- Phase 2: Multi-start Broyden from perturbed points ---
    best_x = x.copy()
    best_norm = np.inf
    try:
        Fx_best = F(best_x)
        if np.all(np.isfinite(Fx_best)):
            best_norm = np.linalg.norm(Fx_best)
    except Exception:
        pass

    if result is not None:
        best_x = result.copy()
        try:
            Fx_best = F(best_x)
            if np.all(np.isfinite(Fx_best)):
                best_norm = np.linalg.norm(Fx_best)
        except Exception:
            pass

    if best_norm <= 1e-8:
        return best_x

    if not time_up():
        for dx in [1e-2, -1e-2, 1e-1, -1e-1, 0.5, -0.5, 1.0, -1.0]:
            if time_up():
                break
            x_try = best_x + np.array([dx, -dx])
            res = _broyden_simple(F, x_try, start_time, deadline)
            if res is not None:
                try:
                    Fx_res = F(res)
                    if np.all(np.isfinite(Fx_res)):
                        norm_res = np.linalg.norm(Fx_res)
                        if norm_res <= 1e-8:
                            return res
                        if norm_res < best_norm:
                            best_norm = norm_res
                            best_x = res.copy()
                except Exception:
                    pass

    # --- Phase 3: Homotopy fallback ---
    if not time_up() and best_norm > 1e-8:
        result = _homotopy_fallback(F, best_x, start_time, deadline)
        if result is not None:
            try:
                Fx_res = F(result)
                if np.all(np.isfinite(Fx_res)):
                    norm_res = np.linalg.norm(Fx_res)
                    if norm_res <= 1e-8:
                        return result
                    if norm_res < best_norm:
                        best_norm = norm_res
                        best_x = result.copy()
            except Exception:
                pass

    # --- Phase 4: SciPy fallback ---
    if not time_up() and best_norm > 1e-8:
        try:
            from scipy.optimize import root as scipy_root
            res = scipy_root(F, best_x, method='hybr', options={'maxfev': 100})
            if np.all(np.isfinite(res.x)):
                Fx_res = F(res.x)
                if np.all(np.isfinite(Fx_res)):
                    norm_res = np.linalg.norm(Fx_res)
                    if norm_res <= 1e-8:
                        return res.x
                    if norm_res < best_norm:
                        best_norm = norm_res
                        best_x = res.x.copy()
        except Exception:
            pass

    return best_x


def _trust_region_dogleg(F, x0, start_time, deadline):
    """
    Trust-region dogleg method with Broyden Hessian approximation.
    Solves min ||F(x)||^2 / 2 subject to ||x - x_k|| <= delta.
    """
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    
    try:
        Fx = F(x)
    except Exception:
        return None
    if not np.all(np.isfinite(Fx)):
        return None
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian
    J = _compute_jacobian(F, x, Fx)
    
    # Initial Hessian approximation: H = J^T J (Gauss-Newton)
    H = J.T @ J
    
    # Trust region radius
    delta = 1.0
    if norm_Fx > 1.0:
        delta = 10.0
    
    best_x = x.copy()
    best_norm = norm_Fx
    
    for iteration in range(40):
        if time_up():
            break
        
        # Solve H d = -g where g = J^T F
        g = J.T @ Fx
        
        # Newton step (unconstrained minimizer)
        try:
            d_newton = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            try:
                d_newton = np.linalg.lstsq(H, -g, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(d_newton)):
            break
        
        norm_newton = np.linalg.norm(d_newton)
        
        # Steepest descent direction
        norm_g = np.linalg.norm(g)
        if norm_g < 1e-15:
            break
        d_steepest = -g / norm_g
        
        # Dogleg path: from 0 to d_steepest to d_newton
        # If ||d_newton|| <= delta, use d_newton
        # Else if ||d_steepest|| >= delta, use delta * d_steepest
        # Else, use d_steepest + t * (d_newton - d_steepest) where t chosen so ||d|| = delta
        
        if norm_newton <= delta:
            d = d_newton
        else:
            # Find intersection with trust region boundary
            # d = d_steepest + t * (d_newton - d_steepest), ||d|| = delta
            v = d_newton - d_steepest
            a = np.dot(v, v)
            b = 2 * np.dot(d_steepest, v)
            c = np.dot(d_steepest, d_steepest) - delta * delta
            
            if a < 1e-15:
                # d_newton and d_steepest are nearly parallel
                d = delta * d_steepest
            else:
                disc = b * b - 4 * a * c
                if disc < 0:
                    d = delta * d_steepest
                else:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b - sqrt_disc) / (2 * a)
                    t2 = (-b + sqrt_disc) / (2 * a)
                    # Choose t in [0, 1]
                    if 0 <= t1 <= 1:
                        t = t1
                    elif 0 <= t2 <= 1:
                        t = t2
                    else:
                        # Use steepest descent scaled to boundary
                        t = 0
                    d = d_steepest + t * v
        
        if not np.all(np.isfinite(d)):
            break
        if np.linalg.norm(d) < 1e-14:
            break
        
        # Evaluate at x + d
        x_new = x + d
        if not np.all(np.isfinite(x_new)):
            delta *= 0.5
            continue
        
        try:
            Fx_new = F(x_new)
        except Exception:
            delta *= 0.5
            continue
        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            continue
        
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        if norm_Fx_new <= 1e-8:
            return x_new
        
        if norm_Fx_new < best_norm:
            best_norm = norm_Fx_new
            best_x = x_new.copy()
        
        # Predicted reduction
        predicted = 0.5 * (np.dot(g, d) + 0.5 * np.dot(d, H @ d))
        actual = 0.5 * (norm_Fx * norm_Fx - norm_Fx_new * norm_Fx_new)
        
        if predicted > 1e-15:
            ratio = actual / predicted
        else:
            ratio = 0.0
        
        if ratio > 0.75:
            # Excellent step, expand trust region
            delta = min(delta * 2.0, 100.0)
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            # Update Jacobian
            J = _compute_jacobian(F, x, Fx)
            # Update Hessian: H = J^T J + Broyden correction
            H = J.T @ J
            # Broyden update for Hessian
            s = x_new - (x - d)  # s = d
            y = J.T @ Fx - g  # gradient change
            if np.dot(s, s) > 1e-15:
                H = H + np.outer(y - H @ s, s) / np.dot(s, s)
        
        elif ratio > 0.25:
            # Good step, keep trust region
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            J = _compute_jacobian(F, x, Fx)
            H = J.T @ J
            s = d
            y = J.T @ Fx - g
            if np.dot(s, s) > 1e-15:
                H = H + np.outer(y - H @ s, s) / np.dot(s, s)
        
        elif ratio > 0.1:
            # Accept step but shrink trust region
            delta *= 0.5
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            J = _compute_jacobian(F, x, Fx)
            H = J.T @ J
            s = d
            y = J.T @ Fx - g
            if np.dot(s, s) > 1e-15:
                H = H + np.outer(y - H @ s, s) / np.dot(s, s)
        
        else:
            # Reject step, shrink trust region
            delta *= 0.5
            if delta < 1e-10:
                break
    
    if norm_Fx <= 1e-8:
        return x
    return best_x if best_norm < 1e-4 else None


def _broyden_simple(F, x0, start_time, deadline):
    """Simple Broyden's method with line search."""
    def time_up():
        return time.time() > deadline

    x = np.array(x0, dtype=float).copy()
    
    try:
        Fx = F(x)
    except Exception:
        return None
    if not np.all(np.isfinite(Fx)):
        return None
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    
    J = _compute_jacobian(F, x, Fx)
    
    best_x = x.copy()
    best_norm = norm_Fx
    
    for iteration in range(30):
        if time_up():
            break
        
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                d = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(d)):
            break
        if np.linalg.norm(d) < 1e-12:
            break
        
        alpha = 1.0
        step_accepted = False
        for ls in range(12):
            if time_up():
                break
            x_new = x + alpha * d
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            try:
                Fx_new = F(x_new)
            except Exception:
                alpha *= 0.5
                continue
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            norm_new = np.linalg.norm(Fx_new)
            if norm_new <= 1e-8:
                return x_new
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new.copy()
            if norm_new < (1.0 - 1e-3 * alpha) * norm_Fx:
                d_dot_d = np.dot(d, d)
                if d_dot_d > 1e-15:
                    delta_F = Fx_new - Fx - J @ d
                    J = J + np.outer(delta_F, d) / d_dot_d
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                step_accepted = True
                break
            alpha *= 0.5
        
        if not step_accepted:
            break
    
    if norm_Fx <= 1e-8:
        return x
    return best_x if best_norm < 1e-4 else None


def _compute_jacobian(F, x, Fx):
    """Compute Jacobian using central differences."""
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_plus = F(x + e)
            Fx_minus = F(x - e)
        except Exception:
            try:
                Fx_plus = F(x + e)
                if not np.all(np.isfinite(Fx_plus)):
                    J[:, j] = 0.0
                else:
                    J[:, j] = (Fx_plus - Fx) / h
                continue
            except Exception:
                J[:, j] = 0.0
                continue
        
        if not (np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus))):
            if not np.all(np.isfinite(Fx_plus)):
                J[:, j] = 0.0
            else:
                J[:, j] = (Fx_plus - Fx) / h
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
    return J


def _homotopy_fallback(F, x_start, start_time, deadline):
    """
    Homotopy/continuation: trace path from anchor to target F.
    H_t(x) = (1-t)*(x - anchor) + t*F(x)
    """
    def time_up():
        return time.time() > deadline

    anchors = [x_start.copy(), x_start + 1.0, x_start - 1.0, x_start + 0.5, x_start - 0.5]
    
    best_x = x_start.copy()
    best_norm = np.inf
    try:
        Fx_best = F(best_x)
        if np.all(np.isfinite(Fx_best)):
            best_norm = np.linalg.norm(Fx_best)
    except Exception:
        pass

    for anchor in anchors:
        if time_up():
            break
        
        x_curr = anchor.copy()
        t_curr = 0.0
        dt = 0.2
        
        for t_step in range(12):
            if time_up():
                break
            
            t_target = min(t_curr + dt, 1.0)
            
            def H_t(x_val, t_val=t_target, anchor_val=anchor):
                return (1 - t_val) * (x_val - anchor_val) + t_val * F(x_val)
            
            try:
                Hx = H_t(x_curr)
            except Exception:
                dt = max(dt * 0.5, 0.05)
                continue
            
            if not np.all(np.isfinite(Hx)):
                dt = max(dt * 0.5, 0.05)
                continue
            
            # Use Broyden to solve H_t(x) = 0
            x_broyden = x_curr.copy()
            Fx_broyden = Hx.copy()
            norm_broyden = np.linalg.norm(Hx)
            
            # Initial Jacobian for H_t
            h = 1e-5
            J_h = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Hp = H_t(x_curr + e)
                    Hm = H_t(x_curr - e)
                    if np.all(np.isfinite(Hp)) and np.all(np.isfinite(Hm)):
                        J_h[:, j] = (Hp - Hm) / (2 * h)
                    else:
                        Hp = H_t(x_curr + e)
                        if np.all(np.isfinite(Hp)):
                            J_h[:, j] = (Hp - Hx) / h
                        else:
                            J_h[:, j] = 0.0
                except Exception:
                    J_h[:, j] = 0.0
            
            converged = False
            for b_iter in range(15):
                if time_up():
                    break
                
                try:
                    d = np.linalg.solve(J_h, -Fx_broyden)
                except np.linalg.LinAlgError:
                    try:
                        d = np.linalg.lstsq(J_h, -Fx_broyden, rcond=None)[0]
                    except Exception:
                        break
                
                if not np.all(np.isfinite(d)):
                    break
                if np.linalg.norm(d) < 1e-12:
                    converged = True
                    break
                
                alpha = 1.0
                step_ok = False
                for ls in range(8):
                    if time_up():
                        break
                    x_trial = x_broyden + alpha * d
                    try:
                        H_trial = H_t(x_trial)
                    except Exception:
                        alpha *= 0.5
                        continue
                    
                    if not np.all(np.isfinite(H_trial)):
                        alpha *= 0.5
                        continue
                    
                    norm_trial = np.linalg.norm(H_trial)
                    if norm_trial <= 1e-10:
                        x_broyden = x_trial
                        Fx_broyden = H_trial
                        norm_broyden = norm_trial
                        converged = True
                        step_ok = True
                        break
                    
                    if norm_trial < (1.0 - 1e-3 * alpha) * norm_broyden:
                        d_dot_d = np.dot(d, d)
                        if d_dot_d > 1e-15:
                            delta_F = H_trial - Fx_broyden - J_h @ d
                            J_h = J_h + np.outer(delta_F, d) / d_dot_d
                        
                        x_broyden = x_trial
                        Fx_broyden = H_trial
                        norm_broyden = norm_trial
                        step_ok = True
                        break
                    alpha *= 0.5
                
                if not step_ok:
                    break
            
            if converged and t_target >= 1.0:
                try:
                    Fx_check = F(x_broyden)
                    if np.all(np.isfinite(Fx_check)):
                        norm_check = np.linalg.norm(Fx_check)
                        if norm_check <= 1e-8:
                            return x_broyden
                        if norm_check < best_norm:
                            best_norm = norm_check
                            best_x = x_broyden.copy()
                except Exception:
                    pass
                break
            elif converged:
                x_curr = x_broyden
                t_curr = t_target
            else:
                dt = max(dt * 0.5, 0.05)

    return best_x if best_norm < np.inf else None

# EVOLVE-BLOCK-END
