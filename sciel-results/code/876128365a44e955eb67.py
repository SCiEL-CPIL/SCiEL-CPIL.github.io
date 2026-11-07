import numpy as np
import time
from scipy.optimize import root

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    Uses Broyden's method for efficiency (1 F call per iter after init).
    Falls back to SciPy's hybr method if Broyden fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    x = np.array(x0, dtype=float)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # Helper to check if valid
    def is_valid(x):
        return np.all(np.isfinite(x))

    # --- Phase 1: Broyden's Method ---
    # Initialize Jacobian with finite differences
    eps = 1e-7
    J = np.zeros((2, 2))
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x # F returned invalid, can't proceed

    # Initial J estimation (3 calls: F(x) already done, need F(x+e0), F(x+e1))
    # We have Fx.
    x0_copy = x.copy()
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        J[:, j] = (F(x + e) - Fx) / eps
    
    # Check initial solution
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Broyden iteration
    # Max iterations to avoid infinite loops
    max_iters = 100 
    calls = 3 # 1 for Fx, 2 for J init
    
    for k in range(max_iters):
        if time_up():
            break
            
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-8:
            return x

        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse or break
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Line search to ensure reduction
        # Try full step first, then shrink
        alpha = 1.0
        found_step = False
        for _ in range(20):
            x_new = x + alpha * dx
            if not is_valid(x_new):
                alpha *= 0.5
                continue
            Fx_new = F(x_new)
            calls += 1
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            # Check for sufficient decrease (Armijo-like condition)
            # ||Fx_new|| < ||Fx||
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                found_step = True
                break
            alpha *= 0.5
            
        if not found_step:
            # If no step reduces norm, try a small random perturbation or break
            # Or accept the step if it's the best we can do?
            # For robustness, if we can't reduce, we might be stuck.
            # Let's try to proceed with the last computed Fx_new if available, else break
            if 'Fx_new' in locals() and np.all(np.isfinite(Fx_new)):
                 # Update x to x_new even if not strictly decreasing? Risky.
                 # Better to break and fallback.
                 pass
            break

        # Update x and Fx
        x = x_new
        Fx = Fx_new
        
        # Broyden Update: J_new = J + (s - J dx) s^T / (s^T s)
        # where s = x_new - x_old, dx is the step taken
        # Note: dx in solve was the full step. The actual step is s = alpha * dx_full
        s = x - (x - alpha * dx) # This is just alpha * dx
        # Wait, x was updated to x_new. 
        # Let's keep track properly.
        # x_old = x_prev
        # s = x_new - x_old
        # y = F(x_new) - F(x_old)
        
        # Re-doing update logic cleanly
        # We need x_old and Fx_old
        # I'll restructure the loop to keep x_old
        
        # Actually, let's just use the standard formula with the values we have.
        # s = alpha * dx
        # y = Fx_new - Fx_old
        # J += np.outer(y - J @ s, s) / np.dot(s, s)
        
        # In the loop above, I didn't store x_old and Fx_old explicitly for the update.
        # Let's restart the Broyden loop structure to be clearer.
        
    # If we broke out of Broyden without converging, or if the loop above was messy, 
    # it's safer to implement a clean Broyden.
    
    # Let's restart with a cleaner implementation.
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x
        
    # Init J
    eps = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        J[:, j] = (F(x + e) - Fx) / eps
    calls = 3
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    for k in range(100):
        if time.time() > deadline:
            break
        if np.linalg.norm(Fx) <= 1e-8:
            return x
            
        # Solve for step
        try:
            dx = np.linalg.solve(J, -Fx)
        except:
            break
            
        # Line search
        alpha = 1.0
        x_new = x
        Fx_new = Fx
        improved = False
        for _ in range(20):
            x_trial = x + alpha * dx
            if not np.all(np.isfinite(x_trial)):
                alpha *= 0.5
                continue
            Fx_trial = F(x_trial)
            calls += 1
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            if np.linalg.norm(Fx_trial) < np.linalg.norm(Fx):
                x_new = x_trial
                Fx_new = Fx_trial
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            break
            
        # Broyden Update
        s = x_new - x
        y = Fx_new - Fx
        denom = np.dot(s, s)
        if denom > 1e-12:
            J = J + np.outer(y - J @ s, s) / denom
            
        x = x_new
        Fx = Fx_new

    # Check if Broyden succeeded
    if np.linalg.norm(Fx) <= 1e-8 and np.all(np.isfinite(x)):
        return x

    # --- Phase 2: Fallback to SciPy ---
    # If Broyden failed, use SciPy's robust solver.
    # This might use more calls, but better a failure than a timeout or high call count from a stuck loop.
    # We limit the call count for SciPy to avoid exceeding 500 if possible, but robustness is key.
    
    def F_wrapper(x):
        return F(x)
        
    try:
        res = root(F_wrapper, x, method='hybr', tol=1e-12, maxfev=50)
        if res.success or np.linalg.norm(res.fun) <= 1e-8:
            return res.x
    except:
        pass
        
    # Last resort: return best x found so far
    return x
