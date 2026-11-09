"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_TOL = 1e-10
_MAX_CALLS_PER_ATTEMPT = 450
_TOTAL_DEADLINE = 145.0

def _powell_solve(F, x0, max_calls=450, tol=1e-10):
    """
    Powell's method for minimizing 0.5 * ||F(x)||^2.
    """
    x = np.array(x0, dtype=float)
    calls = 0
    
    def phi(x_val):
        nonlocal calls
        calls += 1
        Fx = F(x_val)
        if not np.all(np.isfinite(Fx)):
            return 1e100
        return 0.5 * np.dot(Fx, Fx)
    
    # 1D minimization along direction d
    def min1d(d, x, f_x, step=1.0):
        nonlocal calls
        # Try x + step*d and x - step*d
        x1 = x + step * d
        f1 = phi(x1)
        x2 = x - step * d
        f2 = phi(x2)
        
        best_x = x
        best_f = f_x
        
        if f1 < best_f:
            best_x = x1
            best_f = f1
        if f2 < best_f:
            best_x = x2
            best_f = f2
            
        # If we improved, try one more step in the same direction
        if best_x is not x:
            # Determine direction
            if best_x is x1:
                dir_vec = d
            else:
                dir_vec = -d
                
            x3 = best_x + step * dir_vec
            f3 = phi(x3)
            if f3 < best_f:
                best_x = x3
                best_f = f3
                
        return best_x, best_f

    # Initial directions
    d = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    
    f0 = phi(x)
    
    if f0 < 0.5 * tol * tol:
        return x, calls

    max_cycles = 100
    
    for _ in range(max_cycles):
        if calls >= max_calls:
            break
            
        x_start = x.copy()
        f_start = f0
        
        f_drops = []
        
        # Minimize along each direction
        for i in range(len(d)):
            # Adaptive step size
            step = max(1e-3, 0.1 * np.linalg.norm(x))
            if step == 0:
                step = 1e-3
                
            f_before = f0
            x, f0 = min1d(d[i], x, f0, step)
            f_drops.append(f_before - f0)
            
        # Check convergence
        max_drop = max(f_drops)
        if max_drop < 1e-14:
            break
            
        # Update direction set
        i_max = np.argmax(f_drops)
        i_min = np.argmin(f_drops)
        
        # New direction is the vector from start of cycle to end
        new_dir = x - x_start
        if np.linalg.norm(new_dir) > 1e-12:
            d[i_min] = new_dir
        else:
            break
            
        # Optional extra minimization along the new direction (Powell's standard step)
        # This helps with convergence but costs calls.
        if calls < max_calls:
            step = max(1e-3, 0.1 * np.linalg.norm(x))
            if step == 0:
                step = 1e-3
            f_before = f0
            x, f0 = min1d(d[i_min], x, f0, step)
            # We don't strictly need to track this drop for the set update logic 
            # in the next iteration as the set is already updated, but it helps 
            # if we check convergence again.
            
        if f0 < 0.5 * tol * tol:
            break
            
    return x, calls

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + _TOTAL_DEADLINE
    
    x0 = np.array(x0, dtype=float)
    
    # Generate diverse starting points
    candidates = [x0.copy()]
    for scale in [0.5, 2.0, -1.0]:
        candidates.append(x0 * scale)
        
    np.random.seed(42)
    for _ in range(5):
        candidates.append(np.random.randn(2) * 2.0)
    candidates.append(np.array([0.0, 0.0]))
    
    best_x = x0.copy()
    best_norm = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
        if not np.all(np.isfinite(cand)):
            continue
            
        x_res, calls_used = _powell_solve(F, cand, _MAX_CALLS_PER_ATTEMPT)
        
        if np.all(np.isfinite(x_res)):
            F_res = F(x_res)
            if np.all(np.isfinite(F_res)):
                norm_res = np.linalg.norm(F_res)
                if norm_res < best_norm:
                    best_norm = norm_res
                    best_x = x_res.copy()
                    if norm_res <= _TOL:
                        return best_x
                        
    return best_x
# EVOLVE-BLOCK-END
