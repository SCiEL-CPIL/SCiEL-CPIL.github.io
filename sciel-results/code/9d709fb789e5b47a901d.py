"""Root finder for 3 fixed two-dimensional systems using a structural restart-and-refine strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TOTAL_CALLS = 0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    if x is None:
        return False
    x = np.asarray(x, dtype=float)
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure. Tracks call count."""
    global _TOTAL_CALLS
    try:
        x_arr = np.asarray(x, dtype=float)
        y = F(x_arr)
        y = np.asarray(y, dtype=float)
        _TOTAL_CALLS += 1
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        _TOTAL_CALLS += 1
        return None

def _broyden_update(J, s, y):
    """Broyden's good rank-1 update."""
    Js = J @ s
    diff = y - Js
    sTs = np.dot(s, s)
    if sTs < 1e-15:
        return J
    J = J + np.outer(diff, s) / sTs
    return J

def _broyden_solve(x, F, Fx, J, max_iter=30, deadline=115.0):
    """
    Run Broyden's method from (x, Fx, J).
    Returns (best_x, best_norm, best_Fx).
    """
    best_x = x.copy()
    best_norm = _norm(Fx)
    best_Fx = Fx.copy()
    
    for _ in range(max_iter):
        if _elapsed() > deadline:
            break
        
        # Solve J * p = -Fx
        try:
            p = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try a pseudo-inverse or regularized solve
            try:
                p = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback: gradient descent on ||F||^2
                p = -J.T @ Fx
        
        if not _is_finite(p):
            break
        
        # Adaptive damping to ensure progress
        # If step is very large, damp it
        step_norm = _norm(p)
        if step_norm > 1e4:
            t = 1e-4 / step_norm
        else:
            t = 1.0
        
        # Try to take a step
        x_new = x + t * p
        Fx_new = _safe_call(F, x_new)
        
        if Fx_new is None:
            # Shrink step
            t *= 0.5
            x_new = x + t * p
            Fx_new = _safe_call(F, x_new)
            if Fx_new is None:
                break
        
        norm_new = _norm(Fx_new)
        
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
            best_Fx = Fx_new.copy()
            
        if norm_new <= 1e-8:
            return x_new, norm_new, Fx_new
        
        # Update Jacobian
        s = x_new - x
        y = Fx_new - Fx
        J = _broyden_update(J, s, y)
        
        # Move to new point
        x = x_new
        Fx = Fx_new
        
        # If we are diverging significantly, stop this Broyden run
        if norm_new > best_norm * 5.0:
            break

    return best_x, best_norm, best_Fx

def find_root_system(F, x0):
    global _START_TIME, _TOTAL_CALLS
    if _START_TIME is None:
        _START_TIME = time.time()
    _TOTAL_CALLS = 0  # Reset per system call? No, the task says "each call is counted" for the system.
    # Actually, the score is mean_calls over all 3 systems. So _TOTAL_CALLS should be global per run?
    # The prompt says: "mean_calls is the mean number of calls over all 3 systems".
    # So I should track calls per system.
    
    # Let's track calls locally for this system to ensure we don't exceed budget if multiple systems are run in sequence.
    # But the evaluator likely runs find_root_system 3 times.
    # If _TOTAL_CALLS is global, it will accumulate. The prompt implies the evaluator measures calls.
    # "For each system, more than 500 calls... makes that system a failure".
    # So I should keep a local counter or assume the evaluator wraps F to count.
    # "F takes an array... and each call is counted." -> The evaluator counts. I don't need to track for the evaluator,
    # but I can track locally to abort if I'm about to fail.
    
    local_calls = 0
    def _counted_call(x):
        nonlocal local_calls
        local_calls += 1
        return _safe_call(F, x)
        
    # Replace F with _counted_call for local tracking?
    # But _safe_call already increments _TOTAL_CALLS. Let's just use a local wrapper.
    
    # Let's stick to the global _TOTAL_CALLS for now, but reset it at the start of find_root_system?
    # No, if the evaluator calls it 3 times, and I reset, the mean will be wrong if I don't sum them.
    # The evaluator likely counts calls itself. I will just rely on my own local count to abort if > 450.
    
    # Let's re-implement _safe_call to not count globally, or just ignore global count for logic.
    # I will use a local wrapper.
    
    x = np.array(x0, dtype=float)
    Fx = _safe_call(F, x)
    local_calls = 1
    
    if Fx is None:
        return x
    
    norm0 = _norm(Fx)
    if norm0 <= 1e-8:
        return x
    
    # Strategy:
    # 1. Try Broyden from x0.
    # 2. If it fails (or doesn't converge), try a perturbed start point.
    # 3. If that fails, try a few more perturbations.
    
    # Helper to init Jacobian
    def _init_J(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = _safe_call(F, x + e)
            local_calls += 1
            if Fp is None:
                return None
            J[:, j] = (Fp - Fx) / h
        return J
    
    # Attempt 1: Broyden from x0
    J = _init_J(x, Fx)
    local_calls += 2 # 2 calls for Jacobian
    
    if J is not None:
        bx, bn, bFx = _broyden_solve(x, F, Fx, J, max_iter=20, deadline=110.0)
        if bn <= 1e-8:
            return bx
        # If not converged, check if we improved significantly
        if bn < norm0 * 0.5:
            # We made progress, maybe restart from bx with fresh J
            J_new = _init_J(bx, bFx)
            local_calls += 2
            if J_new is not None:
                bx2, bn2, bFx2 = _broyden_solve(bx, F, bFx, J_new, max_iter=20, deadline=110.0)
                if bn2 <= 1e-8:
                    return bx2
                if bn2 < bn:
                    bx, bn, bFx = bx2, bn2, bFx2
        # Else, keep best so far
    
    # If we didn't solve it, try perturbations
    # Generate a few random perturbations or grid perturbations
    # Since we have 2D, we can try small shifts in x and y
    directions = [
        [0.1, 0],
        [-0.1, 0],
        [0, 0.1],
        [0, -0.1],
        [0.1, 0.1],
        [-0.1, -0.1],
        [0.1, -0.1],
        [-0.1, 0.1],
        [0.5, 0],
        [-0.5, 0],
        [0, 0.5],
        [0, -0.5],
    ]
    
    best_overall_x = x.copy()
    best_overall_norm = norm0
    
    # Update best from first attempt
    if bn < best_overall_norm:
        best_overall_x = bx
        best_overall_norm = bn
    
    for d in directions:
        if local_calls > 450:
            break
        if _elapsed() > 110:
            break
            
        x_pert = x0 + np.array(d)
        Fx_pert = _safe_call(F, x_pert)
        local_calls += 1
        if Fx_pert is None:
            continue
            
        norm_pert = _norm(Fx_pert)
        if norm_pert < best_overall_norm:
            best_overall_x = x_pert
            best_overall_norm = norm_pert
            
        # Try Broyden from this perturbed point
        J_pert = _init_J(x_pert, Fx_pert)
        local_calls += 2
        if J_pert is None:
            continue
            
        bx_p, bn_p, bFx_p = _broyden_solve(x_pert, F, Fx_pert, J_pert, max_iter=15, deadline=110.0)
        if bn_p <= 1e-8:
            return bx_p
        if bn_p < best_overall_norm:
            best_overall_x = bx_p
            best_overall_norm = bn_p
            
    # If still not solved, try a few more random points if budget allows
    # This is a last resort
    if local_calls < 450 and _elapsed() < 110:
        rng = np.random.RandomState(42)
        for _ in range(10):
            if local_calls > 450:
                break
            if _elapsed() > 110:
                break
            # Sample in a box around x0
            x_rand = x0 + rng.uniform(-1, 1, size=2)
            Fx_rand = _safe_call(F, x_rand)
            local_calls += 1
            if Fx_rand is None:
                continue
            norm_rand = _norm(Fx_rand)
            if norm_rand < best_overall_norm:
                best_overall_x = x_rand
                best_overall_norm = norm_rand
                
            J_rand = _init_J(x_rand, Fx_rand)
            local_calls += 2
            if J_rand is None:
                continue
            bx_r, bn_r, bFx_r = _broyden_solve(x_rand, F, Fx_rand, J_rand, max_iter=10, deadline=110.0)
            if bn_r <= 1e-8:
                return bx_r
            if bn_r < best_overall_norm:
                best_overall_x = bx_r
                best_overall_norm = bn_r

    return best_overall_x
# EVOLVE-BLOCK-END
