"""Root finder for 3 fixed two-dimensional systems using dogleg trust-region with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _central_jacobian(F, x, calls_ref):
    """Compute Jacobian using central differences. Returns (J, new_calls)."""
    eps = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_p = F(x + e)
        calls_ref[0] += 1
        Fx_m = F(x - e)
        calls_ref[0] += 1
        if _is_finite(Fx_p) and _is_finite(Fx_m):
            J[:, j] = (Fx_p - Fx_m) / (2 * eps)
        elif _is_finite(Fx_p):
            Fx_0 = F(x)
            calls_ref[0] += 1
            J[:, j] = (Fx_p - Fx_0) / eps
        elif _is_finite(Fx_m):
            Fx_0 = F(x)
            calls_ref[0] += 1
            J[:, j] = (Fx_0 - Fx_m) / eps
        else:
            J[:, j] = 1.0  # fallback
    return J


def _dogleg_trust_region(F, x0, max_iter=40, delta_init=1.0):
    """
    Dogleg trust-region method for 2D systems.
    Returns (x_best, calls, best_norm).
    """
    x = np.array(x0, dtype=float)
    calls = [0]
    
    Fx = F(x)
    calls[0] += 1
    if not _is_finite(Fx):
        return x, calls[0], np.inf
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    delta = delta_init
    J = _central_jacobian(F, x, calls)
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x, calls[0], norm_Fx
        
        # Update best
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
        
        # Newton step
        try:
            dx_newton = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular: use steepest descent direction
            dx_newton = -Fx / (norm_Fx + 1e-15)
        
        norm_dx_newton = np.linalg.norm(dx_newton)
        
        # Steepest descent step
        dx_sd = -Fx
        norm_dx_sd = np.linalg.norm(dx_sd)
        if norm_dx_sd < 1e-15:
            break
        dx_sd = dx_sd / norm_dx_sd * min(delta, norm_dx_sd)
        
        # Dogleg: if Newton step is within trust region, use it
        if norm_dx_newton <= delta:
            p = dx_newton
        else:
            # Dogleg: go along steepest descent to intersection, then to Newton
            # Find t such that ||dx_sd * t + (dx_newton - dx_sd*t)|| = delta
            # Simplified: use a point on the dogleg
            # t is the distance along steepest descent before turning
            # For 2D, compute intersection
            a = dx_sd
            b = dx_newton - dx_sd
            
            # Solve ||a + t*b|| = delta for t in [0, 1]
            # ||a||^2 + 2t(a.b) + t^2||b||^2 = delta^2
            aa = np.dot(a, a)
            ab = np.dot(a, b)
            bb = np.dot(b, b)
            
            if bb < 1e-15:
                p = a
            else:
                # Quadratic: bb*t^2 + 2*ab*t + (aa - delta^2) = 0
                disc = 4*ab*ab - 4*bb*(aa - delta*delta)
                if disc < 0:
                    p = a
                else:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-2*ab + sqrt_disc) / (2*bb)
                    t2 = (-2*ab - sqrt_disc) / (2*bb)
                    # Choose t in [0,1]
                    t = None
                    for tc in [t1, t2]:
                        if 0 <= tc <= 1:
                            t = tc
                            break
                    if t is None:
                        # Use the one closer to [0,1]
                        if abs(t1 - 0.5) < abs(t2 - 0.5):
                            t = np.clip(t1, 0, 1)
                        else:
                            t = np.clip(t2, 0, 1)
                    p = a + t * b
        
        # Evaluate at x + p
        x_new = x + p
        Fx_new = F(x_new)
        calls[0] += 1
        
        if not _is_finite(Fx_new):
            delta *= 0.5
            if delta < 1e-10:
                break
            continue
        
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        # Predicted reduction
        pred_red = -np.dot(Fx, p) - 0.5 * np.dot(p, J @ p)
        actual_red = norm_Fx**2 - norm_Fx_new**2
        
        rho = actual_red / max(pred_red, 1e-15) if pred_red > 0 else 0
        
        if rho > 0.75:
            # Good step, expand trust region
            delta = min(2 * delta, 10.0)
        elif rho > 0.25:
            # Acceptable step
            pass
        else:
            # Poor step, shrink trust region
            delta *= 0.5
            if delta < 1e-10:
                break
            continue
        
        # Accept step
        x = x_new
        Fx = Fx_new
        
        # Update Jacobian using central differences
        if it % 3 == 0 or norm_Fx_new > 0.1 * norm_Fx:
            J = _central_jacobian(F, x, calls)
        else:
            # Broyden-like update as cheap alternative
            s = x_new - (x - p)  # This is just p
            y = Fx_new - (Fx - J @ p)  # Not quite right; let's use proper update
            # Actually, let's just recompute every few iterations for simplicity
            J = _central_jacobian(F, x, calls)
    
    return best_x, calls[0], best_norm


def _generate_starts(x0, n_starts=5):
    """Generate multiple starting points around x0."""
    starts = [np.array(x0, dtype=float)]
    rng = np.random.default_rng(42)
    for i in range(n_starts - 1):
        # Scale based on distance from origin or fixed scale
        offset = rng.normal(0, 1.0, size=2)
        starts.append(np.array(x0, dtype=float) + offset)
    return starts


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses dogleg trust-region method with multi-start.
    """
    x0 = np.array(x0, dtype=float)
    
    # Generate multiple starting points
    starts = _generate_starts(x0, n_starts=4)
    
    best_x = None
    best_norm = np.inf
    total_calls = 0
    
    for i, x_start in enumerate(starts):
        if not _time_ok():
            break
        
        # Allow more iterations for the first start (original x0)
        max_iter = 50 if i == 0 else 30
        delta_init = 1.0 if i == 0 else 0.5
        
        x_sol, calls, norm_sol = _dogleg_trust_region(F, x_start, max_iter=max_iter, delta_init=delta_init)
        total_calls += calls
        
        if _is_finite(x_sol) and norm_sol < best_norm:
            best_norm = norm_sol
            best_x = x_sol.copy()
        
        # If we've converged well, stop early
        if best_norm <= 1e-10:
            break
    
    # Final verification and return
    if best_x is not None and _is_finite(best_x):
        return best_x
    else:
        # Return x0 as last resort
        return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
