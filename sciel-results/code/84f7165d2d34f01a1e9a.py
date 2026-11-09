"""Root finder for 3 fixed two-dimensional systems using robust Broyden's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x, call_count):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    x0 = np.array(x0, dtype=float)
    call_count = [0]
    
    def F_counted(x):
        call_count[0] += 1
        return F(x)

    best_x = x0.copy()
    best_norm = np.inf
    
    # Try to get a good initial Jacobian estimate if possible, 
    # but Broyden can start with Identity. 
    # Let's use Finite Difference for the first restart to ensure a good start.
    
    max_restarts = 20
    max_broyden_iter = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline or call_count[0] >= 480:
            break
        
        # Determine starting point
        if restart == 0:
            x_start = x0.copy()
        else:
            # Random restart around best_x
            # Scale perturbation based on distance from origin or just fixed
            scale = 0.5 * (1.0 + np.linalg.norm(best_x))
            angle = np.random.uniform(0, 2 * np.pi)
            radius = scale * (0.1 + 0.9 * np.random.rand())
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
            
        # Evaluate F at start
        Fx = _safe_F(F_counted, x_start, call_count)
        if Fx is None:
            continue
            
        norm_F = np.linalg.norm(Fx)
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x_start.copy()
            
        if norm_F <= 1e-8:
            return best_x
        
        # Initialize Jacobian
        # Use Finite Difference for robustness
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = _safe_F(F_counted, x_start + e, call_count)
            if Fx_p is None:
                # Fallback to identity if FD fails
                J = np.eye(2)
                break
            J[:, j] = (Fx_p - Fx) / h
        else:
            # If loop completes, J is filled. 
            # Check if J is singular or bad
            try:
                np.linalg.inv(J)
            except np.linalg.LinAlgError:
                J = np.eye(2)
                
        if Fx_p is None:
            continue
            
        # Broyden's Method Loop
        x_curr = x_start.copy()
        Fx_curr = Fx
        J_curr = J.copy()
        norm_curr = norm_F
        
        for _ in range(max_broyden_iter):
            if time.time() > deadline or call_count[0] >= 480:
                break
                
            if norm_curr <= 1e-8:
                best_x = x_curr.copy()
                best_norm = norm_curr
                return best_x
                
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J_curr, -Fx_curr)
            except np.linalg.LinAlgError:
                # If singular, reset Jacobian to Identity and retry? Or break?
                # Let's break this Broyden run and restart
                break
                
            # Line Search (Backtracking)
            # We want ||F(x + alpha*dx)|| < ||F(x)||
            alpha = 1.0
            x_new = None
            F_new = None
            norm_new = np.inf
            
            for _ls in range(20):
                x_trial = x_curr + alpha * dx
                F_trial = _safe_F(F_counted, x_trial, call_count)
                if F_trial is None:
                    alpha /= 2.0
                    continue
                
                norm_trial = np.linalg.norm(F_trial)
                
                if norm_trial < norm_curr:
                    x_new = x_trial
                    F_new = F_trial
                    norm_new = norm_trial
                    break
                else:
                    alpha /= 2.0
                    if alpha < 1e-12:
                        break
                        
            if x_new is None:
                # Line search failed, break
                break
                
            # Update Broyden Jacobian
            s = x_new - x_curr
            y = F_new - Fx_curr
            denom = np.dot(s, s)
            if denom > 1e-14:
                # Broyden's good method
                J_curr = J_curr + np.outer(y - J_curr @ s, s) / denom
                
            # Update state
            x_curr = x_new
            Fx_curr = F_new
            norm_curr = norm_new
            
            if norm_curr < best_norm:
                best_norm = norm_curr
                best_x = x_curr.copy()
                
    return best_x
# EVOLVE-BLOCK-END
