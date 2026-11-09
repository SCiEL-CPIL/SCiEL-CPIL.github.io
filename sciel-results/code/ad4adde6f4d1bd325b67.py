import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using an efficient Broyden's method.
    Minimizes function calls by using forward differences for initial Jacobian
    and rank-1 updates thereafter.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 140.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(x_val)
            result = np.asarray(result, dtype=float)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except:
            return None

    # 1. Initial Evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # 2. Initial Jacobian using Forward Differences (3 calls total so far: Fx, Fx+e1, Fx+e2)
    # Using forward differences saves 2 calls compared to central differences.
    # Accuracy is O(h), which is sufficient for Broyden to start.
    h = 1e-6
    J = np.zeros((2, 2))
    
    # We already have Fx.
    # Calculate F(x + h*e_0)
    x_p0 = x.copy()
    x_p0[0] += h
    Fx_p0 = _call_F(x_p0)
    if Fx_p0 is None:
        # Fallback: try central if forward failed? Or just return best.
        # If forward failed, the function might be weird.
        return best_x
        
    J[:, 0] = (Fx_p0 - Fx) / h

    # Calculate F(x + h*e_1)
    x_p1 = x.copy()
    x_p1[1] += h
    Fx_p1 = _call_F(x_p1)
    if Fx_p1 is None:
        return best_x
        
    J[:, 1] = (Fx_p1 - Fx) / h

    # 3. Broyden's Good Method Iterations
    # We have used 3 calls. Budget is ~495. ~490 iterations possible.
    stagnation_count = 0
    prev_norm = norm_Fx
    
    for it in range(450):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Check convergence
        if norm_Fx <= 1e-8:
            return x

        # Solve J @ s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, perturb or use pinv
            try:
                s = np.linalg.pinv(J) @ (-Fx)
            except:
                break

        # Step size control
        norm_s = np.linalg.norm(s)
        if norm_s == 0:
            break
            
        # Cap step size to prevent wild jumps
        max_step = 1.0 
        if norm_s > max_step:
            s = s * (max_step / norm_s)
            norm_s = max_step

        x_new = x + s
        
        # Check finite
        if not np.all(np.isfinite(x_new)):
            # Try smaller step
            s = s * 0.1
            x_new = x + s
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
            
        norm_Fx_new = np.linalg.norm(Fx_new)

        # Track best
        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        # Convergence check
        if norm_Fx_new <= 1e-8:
            return x_new

        # Stagnation check
        if norm_Fx_new > prev_norm * 0.99: # Not decreasing significantly
            stagnation_count += 1
        else:
            stagnation_count = 0
            
        # If stagnant, try a fresh Jacobian (expensive but breaks deadlock)
        if stagnation_count >= 5:
            # Refresh Jacobian at current best_x
            x = best_x.copy()
            Fx = _call_F(x)
            if Fx is None:
                break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x
                
            # Recalculate J with forward diffs
            h = 1e-6
            J = np.zeros((2, 2))
            x_p0 = x.copy(); x_p0[0] += h
            Fx_p0 = _call_F(x_p0)
            if Fx_p0 is not None:
                J[:, 0] = (Fx_p0 - Fx) / h
            else:
                break
            x_p1 = x.copy(); x_p1[1] += h
            Fx_p1 = _call_F(x_p1)
            if Fx_p1 is not None:
                J[:, 1] = (Fx_p1 - Fx) / h
            else:
                break
                
            stagnation_count = 0
            prev_norm = norm_Fx
            continue

        # Broyden Update
        y = Fx_new - Fx
        # s is the step taken
        # J_new = J + (y - J @ s) * s^T / (s^T @ s)
        denom = np.dot(s, s)
        if denom > 1e-15:
            J += np.outer(y - J @ s, s) / denom

        x = x_new
        Fx = Fx_new
        norm_Fx = norm_Fx_new
        prev_norm = norm_Fx

    return best_x
