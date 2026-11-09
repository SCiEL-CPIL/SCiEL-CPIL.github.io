"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Safe F wrapper
    def safe_F(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            return None

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return x
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian with forward differences (2 calls)
    h = 1e-7
    J = np.zeros((2, 2))
    valid_J = True
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = safe_F(x + e)
        if Fx_pert is None:
            valid_J = False
            break
        J[:, j] = (Fx_pert - Fx) / h
        
    if not valid_J or np.linalg.cond(J) > 1e15:
        J = np.eye(2)

    calls_used = 3

    # Broyden's Method
    for i in range(200):
        if time.time() > deadline:
            break
        if calls_used >= 490:
            break

        # Compute step
        try:
            step = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use identity or gradient descent step
            J = np.eye(2)
            step = -Fx

        # Damping to prevent huge steps
        step_norm = np.linalg.norm(step)
        if step_norm > 1e5:
            step = step * (1e5 / step_norm)
            
        x_new = x + step
        Fx_new = safe_F(x_new)
        calls_used += 1

        if Fx_new is None:
            # Revert and try smaller step
            step = step * 0.1
            x_new = x + step
            Fx_new = safe_F(x_new)
            calls_used += 1
            if Fx_new is None:
                # Give up on this direction, reset J
                J = np.eye(2)
                continue

        if np.linalg.norm(Fx_new) <= 1e-8:
            return x_new

        # Broyden Update
        s = x_new - x
        y = Fx_new - Fx
        denom = np.dot(s, s)
        
        if denom > 1e-14:
            J = J + np.outer(y - J @ s, s) / denom
        else:
            # Stagnation
            # Perturb to escape
            x_new = x + np.random.normal(0, 1e-4, 2)
            Fx_new = safe_F(x_new)
            calls_used += 1
            if Fx_new is None:
                break
            J = np.eye(2)

        x = x_new
        Fx = Fx_new

    return x
# EVOLVE-BLOCK-END
