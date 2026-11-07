# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0 using a Secant-Newton hybrid with safeguards.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s)
    tol = 1e-8
    
    x = float(x0)
    h = 1e-4 * max(1.0, abs(x))
    x_next = x + h
    
    try:
        f_curr = f(x)
        f_next = f(x_next)
    except Exception:
        return x

    if not math.isfinite(f_curr) or not math.isfinite(f_next):
        return x

    if abs(f_curr) <= tol:
        return x
    if abs(f_next) <= tol:
        return x_next

    best_x = x if abs(f_curr) < abs(f_next) else x_next
    best_f = min(abs(f_curr), abs(f_next))
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        if f_curr == f_next:
            # Degenerate case, perturb x_next
            x_next = x_next + 1e-4 * max(1.0, abs(x_next))
            try:
                f_next = f(x_next)
            except Exception:
                break
            if not math.isfinite(f_next):
                break
            if abs(f_next) <= tol:
                return x_next
            if abs(f_next) < best_f:
                best_x = x_next
                best_f = abs(f_next)
            continue

        x_new = x_next - f_next * (x_next - x) / (f_next - f_curr)
        
        if not math.isfinite(x_new):
            break
        
        # Safeguard against divergence
        if abs(x_new - x_next) > 10 * max(1.0, abs(x_next)):
            # Try a damped Newton step using the secant slope as derivative estimate
            slope = (f_next - f_curr) / (x_next - x)
            if slope != 0 and math.isfinite(slope):
                x_new_damped = x_next - f_next / slope
                if math.isfinite(x_new_damped):
                    x_new = x_new_damped
                else:
                    break
            else:
                break

        try:
            f_new = f(x_new)
        except Exception:
            break
        
        if not math.isfinite(f_new):
            break

        if abs(f_new) <= tol:
            return x_new

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)
            
        x, f_curr = x_next, f_next
        x_next, f_next = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
