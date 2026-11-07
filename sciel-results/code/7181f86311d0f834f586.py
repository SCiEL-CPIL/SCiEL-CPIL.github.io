# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's quasi-Newton method
    with backtracking line search and restarts.
    """
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin before 120s kill

    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    def safe_eval_F(v):
        """Evaluate F with safety checks, returning (Fx, success)."""
        try:
            Fx = F(v)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, False
            return Fx, True
        except Exception:
            return None, False

    def broyden_solver(x_start, max_iter=100, call_budget=450):
        """
        Run Broyden's method from x_start.
        Returns (x_best, best_norm, calls_used).
        """
        x = np.array(x_start, dtype=float).copy()
        calls = 0
        
        # Initial evaluation
        Fx, ok = safe_eval_F(x)
        if not ok:
            return x, np.inf, calls
        calls += 1
        
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm <= 1e-8:
            return x, Fx_norm, calls

        # Initialize Jacobian estimate using finite differences
        J = np.zeros((2, 2))
        h = 1e-6
        for i in range(2):
            if calls >= call_budget or not time_ok():
                break
            xp = x.copy()
            xp[i] += h
            Fxp, okp = safe_eval_F(xp)
            if okp:
                J[:, i] = (Fxp - Fx) / h
            else:
                J[:, i] = np.eye(2)[:, i]
            calls += 1
        
        if not np.all(np.isfinite(J)):
            J = np.eye(2)

        best_x = x.copy()
        best_norm = Fx_norm

        for _ in range(max_iter):
            if calls >= call_budget or not time_ok():
                break

            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            # Backtracking line search
            alpha = 1.0
            step_accepted = False
            for _ in range(10):
                x_new = x + alpha * dx
                Fx_new, ok_new = safe_eval_F(x_new)
                if not ok_new:
                    alpha *= 0.5
                    continue
                calls += 1
                
                Fx_new_norm = np.linalg.norm(Fx_new)
                
                if Fx_new_norm <= 1e-8:
                    return x_new, Fx_new_norm, calls

                # Accept if we reduce the residual
                if Fx_new_norm < Fx_norm:
                    step_accepted = True
                    break
                alpha *= 0.5
            
            if not step_accepted:
                # If no step accepted, try a smaller step in a random direction? 
                # Or just break if we can't improve.
                # For now, break to save calls.
                break

            # Update Broyden's Jacobian (Good variant)
            s = x_new - x
            y = Fx_new - Fx
            s_norm_sq = s @ s
            if s_norm_sq > 1e-15:
                J = J + (np.outer(y, s) - np.outer(J @ s, s)) / s_norm_sq

            x = x_new
            Fx = Fx_new
            Fx_norm = Fx_new_norm

            if Fx_norm < best_norm:
                best_x = x_new.copy()
                best_norm = Fx_norm

        return best_x, best_norm, calls

    x = np.array(x0, dtype=float).copy()
    
    # Check if initial point is valid
    Fx_init, ok_init = safe_eval_F(x)
    if not ok_init:
        # Local search to find a valid point
        x = _local_search(F, x, time_ok)
        Fx_init, ok_init = safe_eval_F(x)
        if not ok_init:
            return x

    # First attempt from x0
    x_best, norm_best, calls_used = broyden_solver(x, max_iter=60, call_budget=400)
    
    if norm_best <= 1e-8:
        return x_best

    # If failed, try restarts
    # Generate restart points: scaled x0, perturbations
    restarts = []
    x0_arr = np.array(x0, dtype=float)
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0]:
        restarts.append(x0_arr * scale)
    for axis in range(2):
        for delta in [-1.0, 1.0, -0.1, 0.1]:
            pt = x0_arr.copy()
            pt[axis] += delta
            restarts.append(pt)

    remaining_budget = 480 - calls_used
    if remaining_budget <= 10:
        return x_best

    for r_pt in restarts:
        if not time_ok():
            break
        # Allocate a portion of remaining budget to each restart
        alloc = min(remaining_budget - 10, 100) 
        if alloc < 10:
            break
            
        x_r, norm_r, calls_r = broyden_solver(r_pt, max_iter=40, call_budget=alloc)
        calls_used += calls_r
        remaining_budget -= calls_r
        
        if norm_r < norm_best:
            x_best = x_r
            norm_best = norm_r
            
        if norm_best <= 1e-8:
            return x_best

    return x_best

def _local_search(F, x, time_ok):
    """Simple local search to find a point where F is finite."""
    x = np.array(x, dtype=float)
    for dx in [1e-2, 1e-1, 1.0, 10.0]:
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            if not time_ok():
                return x
            step = dx * np.array([np.cos(angle), np.sin(angle)])
            x_try = x + step
            try:
                Fx = F(x_try)
                Fx = np.asarray(Fx, dtype=float)
                if Fx.shape != (2,):
                    Fx = Fx.reshape(2)
                if np.all(np.isfinite(Fx)):
                    x = x_try
                    return x
            except Exception:
                continue
    return x
# EVOLVE-BLOCK-END
