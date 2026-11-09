import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 115.0  # Safety margin from 120s limit
    
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        # If initial point is invalid, try a small perturbation
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-8:
        return x

    # Homotopy Continuation Method
    # We define a homotopy H(x, t) = (1-t)*x + t*F(x)
    # At t=0, the root is x=0.
    # At t=1, the root is the solution to F(x)=0.
    # We track the path from t=0 to t=1.
    
    # Start at t=0, x=0
    x = np.zeros(2)
    t = 0.0
    
    # We will increase t in steps. 
    # At each step, we solve for the new x such that H(x, t_new) = 0.
    # This is a nonlinear system in x for a fixed t.
    # We can use Newton's method to solve H(x, t) = 0 for x, given a good initial guess (previous x).
    # The Jacobian of H with respect to x is J_H = (1-t)*I + t*J_F.
    # We approximate J_F using finite differences or previous estimates.
    
    # To be efficient and robust, we will use a predictor-corrector scheme.
    # Predictor: x_pred = x_current
    # Corrector: Solve H(x, t_new) = 0 starting from x_pred.
    
    # Number of homotopy steps
    # We want to balance accuracy and number of calls.
    # Let's try 20-50 steps.
    n_steps = 30
    t_steps = np.linspace(0, 1, n_steps + 1)[1:] # Exclude 0
    
    best_x = x.copy()
    best_norm = _norm(Fx)
    
    # We need to estimate the Jacobian of F at the current point to compute J_H.
    # However, at t=0, we don't need J_F.
    # As t increases, we need better estimates.
    # Let's maintain an estimate of J_F.
    
    # Initial J_F estimate at x=0? Or at first valid point?
    # Actually, for the first step from t=0 to t=dt, we can just try to solve.
    # But H(x, t) = (1-t)x + t F(x).
    # At t=0, H(x,0)=x. Root is 0.
    # At t=dt, H(x, dt) = (1-dt)x + dt F(x) = 0.
    
    # Let's use a simple Newton solver inside the homotopy loop.
    # This might be expensive in calls if not careful.
    # Alternative: Use the fact that for small t, the root is close to 0.
    
    # Let's refine:
    # We will iterate t from 0 to 1.
    # For each t, we solve for x.
    # We use Newton's method on G_t(x) = (1-t)x + t F(x) = 0.
    # J_G = (1-t)I + t J_F.
    # We need J_F. We will compute it via finite differences only when necessary or maintain an approximation.
    # Given the 500 call limit, we can't afford FD at every Newton step of every homotopy step.
    # 
    # Revised Strategy:
    # 1. Compute J_F at x=0 (cost 3 calls: F(0), F(e1), F(e2)).
    # 2. Use this J_F to predict the path.
    # 3. As we move, update J_F using Broyden updates based on the homotopy steps?
    #    Or just re-compute FD periodically?
    #    
    # Let's stick to a robust "Predictor-Corrector" with limited Newton steps.
    
    # Compute initial J_F at x=0
    h = 1e-5
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    F0 = _safe_call(F, np.zeros(2))
    if F0 is None:
        return x # F(0) is invalid, homotopy from 0 is bad. Fallback to x0?
        
    Fe1 = _safe_call(F, e1)
    Fe2 = _safe_call(F, e2)
    
    if Fe1 is None or Fe2 is None:
        # Fallback to simple Broyden from x0
        # (Re-using logic from previous strategy as fallback)
        x = np.array(x0, dtype=float)
        Fx = _safe_call(F, x)
        if Fx is None: return x
        if _norm(Fx) <= 1e-8: return x
        
        # Simple Broyden
        J = np.eye(2) # Assume identity if FD fails
        x_old = x
        Fx_old = Fx
        for _ in range(20):
            if _elapsed() > deadline: break
            try:
                d = np.linalg.solve(J, -Fx)
            except:
                break
            if not np.all(np.isfinite(d)): break
            x_new = x + d
            Fx_new = _safe_call(F, x_new)
            if Fx_new is None: break
            if _norm(Fx_new) < _norm(Fx):
                s = x_new - x
                y = Fx_new - Fx
                J = J + np.outer(y - J@s, s)/(s@s + 1e-16)
                x = x_new
                Fx = Fx_new
                if _norm(Fx) <= 1e-8: break
                x_old = x
                Fx_old = Fx
            else:
                break
        return x

    J_F = np.zeros((2, 2))
    J_F[:, 0] = (Fe1 - F0) / h
    J_F[:, 1] = (Fe2 - F0) / h
    
    # Start homotopy
    x = np.zeros(2)
    t = 0.0
    
    # We will take steps in t.
    # Adaptive step size: start with 0.1, reduce if Newton fails.
    current_t = 0.0
    max_t = 1.0
    dt = 0.1
    
    while current_t < max_t - 1e-12:
        if _elapsed() > deadline:
            break
            
        next_t = min(current_t + dt, max_t)
        
        # Solve H(x, next_t) = 0 for x, starting from current x
        # Newton iterations on G(x) = (1-next_t)*x + next_t*F(x)
        
        success = False
        for _newton_iter in range(10):
            if _elapsed() > deadline:
                break
                
            # Evaluate G(x)
            Fx = _safe_call(F, x)
            if Fx is None:
                break
                
            G = (1 - next_t) * x + next_t * Fx
            norm_G = _norm(G)
            
            if norm_G <= 1e-12:
                success = True
                break
                
            # Compute J_G = (1-next_t)*I + next_t*J_F
            # We use the current J_F estimate.
            J_G = (1 - next_t) * np.eye(2) + next_t * J_F
            
            try:
                d = np.linalg.solve(J_G, -G)
            except np.linalg.LinAlgError:
                # Regularize
                J_G_reg = J_G + 1e-8 * np.eye(2)
                try:
                    d = np.linalg.solve(J_G_reg, -G)
                except:
                    break
                    
            if not np.all(np.isfinite(d)):
                break
                
            # Damping: ensure step is not too large
            # Check if step reduces ||G||
            x_trial = x + d
            Fx_trial = _safe_call(F, x_trial)
            if Fx_trial is None:
                d *= 0.5
                continue
                
            G_trial = (1 - next_t) * x_trial + next_t * Fx_trial
            norm_G_trial = _norm(G_trial)
            
            if norm_G_trial < norm_G:
                # Update J_F using Broyden update based on F change
                s = x_trial - x
                y = Fx_trial - Fx
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-16:
                    J_F = J_F + np.outer(y - J_F @ s, s) / s_dot_s
                    
                x = x_trial
                # Continue Newton iterations for this t
            else:
                # Reduce step
                d *= 0.5
                if _norm(d) < 1e-10:
                    break
        
        if success:
            current_t = next_t
            # Reset step size or keep it
            dt = min(dt * 1.2, 0.2) # Slowly increase step size
        else:
            # Failed to solve at next_t. Reduce step size and retry
            dt *= 0.5
            if dt < 1e-4:
                # Stuck, break homotopy
                break
            # Keep current_t same, try smaller dt
            continue
            
        # Check if we reached the end
        if current_t >= max_t - 1e-12:
            break
            
    # Final check
    Fx_final = _safe_call(F, x)
    if Fx_final is not None and _norm(Fx_final) <= 1e-8:
        return x
        
    # If homotopy didn't converge to 1e-8, try polishing from current x with Broyden
    # Re-use Broyden logic as a final polish
    x_old = x
    Fx_old = Fx_final if Fx_final is not None else _safe_call(F, x)
    if Fx_old is None:
        return x
        
    for _ in range(20):
        if _elapsed() > deadline:
            break
        try:
            d = np.linalg.solve(J_F, -Fx_old)
        except:
            break
        if not np.all(np.isfinite(d)):
            break
        x_new = x_old + d
        Fx_new = _safe_call(F, x_new)
        if Fx_new is None:
            break
        if _norm(Fx_new) < _norm(Fx_old):
            s = x_new - x_old
            y = Fx_new - Fx_old
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J_F = J_F + np.outer(y - J_F @ s, s) / s_dot_s
            x_old = x_new
            Fx_old = Fx_new
            if _norm(Fx_old) <= 1e-8:
                break
        else:
            break
            
    return x_old
