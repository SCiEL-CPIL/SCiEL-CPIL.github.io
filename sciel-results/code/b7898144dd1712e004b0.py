"""
Strategy: Deterministic Gradient-Based Optimization (L-BFGS-B)
Instead of random walks (SA), we define a smooth objective function that penalizes
overlaps and boundary violations quadratically. We minimize this objective using
SciPy's L-BFGS-B, which leverages gradient information to navigate the search space
efficiently. We use multiple random restarts to escape local minima.
Finally, we perform a binary search on a global scale factor to maximize the sum of
radii while maintaining feasibility.
"""

import numpy as np
import time
from scipy.optimize import minimize
import random

N = 26

def _soft_objective(params, n):
    """
    Objective function: Minimize negative sum of radii + soft penalties for violations.
    params: flat array of shape (3*n,) [x0, y0, r0, x1, y1, r1, ...]
    """
    xy = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Clip radii to be positive for calculation, but let optimizer handle bounds
    r_safe = np.maximum(r, 1e-6)
    
    # 1. Objective: Maximize sum of radii (so minimize negative sum)
    score = -np.sum(r)
    
    # 2. Penalty for boundary violations
    # x - r < 0 => violation
    # (1-x) - r < 0 => violation
    # We use smooth penalty: max(0, violation)^2
    pen_left = np.maximum(0.0, r_safe - xy)
    pen_right = np.maximum(0.0, r_safe - (1.0 - xy))
    pen_bottom = np.maximum(0.0, r_safe - y)
    pen_top = np.maximum(0.0, r_safe - (1.0 - y))
    
    boundary_penalty = np.sum(pen_left**2) + np.sum(pen_right**2) + np.sum(pen_bottom**2) + np.sum(pen_top**2)
    
    # 3. Penalty for overlaps
    # dist(i,j) < r_i + r_j => violation
    # Smooth penalty: max(0, r_i + r_j - dist)^2
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i] - xy[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            violation = r_safe[i] + r_safe[j] - dist
            if violation > 0:
                overlap_penalty += violation**2
    
    # Weight penalties heavily
    weight = 1000.0
    total = score + weight * (boundary_penalty + overlap_penalty)
    
    return total

def _soft_gradient(params, n):
    """
    Numerical gradient calculation for L-BFGS-B.
    """
    eps = 1e-5
    grad = np.zeros_like(params)
    base_val = _soft_objective(params, n)
    
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        val_plus = _soft_objective(params, n)
        params[i] = orig
        grad[i] = (val_plus - base_val) / eps
        
    return grad

def _is_valid(xy, r, tol=1e-8):
    """
    Strict validity check.
    """
    n = len(xy)
    for i in range(n):
        if xy[i] - r[i] < -tol or 1.0 - xy[i] - r[i] < -tol or \
           xy[i] < -tol or 1.0 - xy[i] < -tol:
            return False
        if xy[i] - r[i] < -tol or 1.0 - xy[i] - r[i] < -tol: # y checks
            pass # Wait, xy is x, y is separate in objective but here xy is array? 
            # Let's assume xy is (N,2) and r is (N,) for clarity in this function
            # Actually, let's stick to 1D arrays for consistency with optimizer
            # But _is_valid is called with unpacked arrays. Let's make it robust.
    
    # Re-implementing for 1D arrays x, y, r
    x = xy[:, 0] if xy.ndim == 2 else xy
    y = xy[:, 1] if xy.ndim == 2 else xy
    
    # Walls
    if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol): return False
    if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol): return False
    
    # Overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _maximize_radii(xy, r, time_limit=5.0):
    """
    Given a valid configuration, try to increase radii using binary search on a scale factor.
    We assume the relative positions are fixed, but we can also allow small adjustments?
    For simplicity and robustness, we just scale radii. If it fails, we return original.
    Actually, a better approach: use the optimizer again with fixed positions?
    No, let's just do a simple binary search on a global multiplier for radii.
    """
    start_time = time.time()
    best_r = r.copy()
    best_val = np.sum(r)
    
    # First, try to expand slightly
    # We will binary search a factor f in [1.0, 2.0]
    lo, hi = 1.0, 2.0
    
    # Check if we can even expand by 1.01
    # If not, return original
    test_r = r * 1.01
    if not _is_valid(xy, test_r):
        return r
    
    # Binary search
    for _ in range(30):
        if time.time() - start_time > time_limit:
            break
        mid = (lo + hi) / 2.0
        test_r = r * mid
        if _is_valid(xy, test_r):
            lo = mid
            best_r = test_r
            best_val = np.sum(test_r)
        else:
            hi = mid
            
    return best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Multiple restarts
    num_restarts = 15
    
    for k in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Initialize random positions
        # Start with small radii to avoid initial overlaps
        x = np.random.rand(N) * 0.8 + 0.1
        y = np.random.rand(N) * 0.8 + 0.1
        r = np.full(N, 0.05)
        
        params = np.concatenate([x, y, r])
        
        # Bounds for L-BFGS-B
        # x, y in [0, 1]
        # r in [0.01, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.01, 0.5)) # r
            
        # Optimize
        try:
            res = minimize(
                _soft_objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                jac=None, # Let it do numerical differentiation, or provide _soft_gradient?
                # Providing gradient is faster but _soft_gradient is O(N^3) due to loop.
                # For N=26, numerical diff is 78 calls. Each call is O(N^2)=676. Total ~53k ops.
                # L-BFGS-B might do 100-500 iterations. 500 * 53k = 26M ops. Fast enough in Python?
                # Maybe 1-2 seconds per restart. 15 restarts = 30s. Feasible.
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
        except Exception:
            continue
            
        opt_params = res.x
        opt_x = opt_params[0::3]
        opt_y = opt_params[1::3]
        opt_r = opt_params[2::3]
        
        # Check validity
        # The optimizer minimizes penalties, so if score is good, it should be valid.
        # But due to numerical issues, it might be slightly invalid.
        # We check strict validity.
        
        # If invalid, try to fix by shrinking radii slightly or skipping
        if _is_valid(opt_x, opt_y, opt_r):
            # Try to maximize radii
            max_r = _maximize_radii(np.column_stack([opt_x, opt_y]), opt_r, time_limit=2.0)
            current_score = np.sum(max_r)
            
            if current_score > best_score:
                best_score = current_score
                best_result = np.column_stack([opt_x, opt_y, max_r])
        else:
            # If slightly invalid, try shrinking radii by 5% and re-checking
            shrunk_r = opt_r * 0.95
            if _is_valid(opt_x, opt_y, shrunk_r):
                current_score = np.sum(shrunk_r)
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack([opt_x, opt_y, shrunk_r])

    # Fallback if no valid result found
    if best_result is None:
        # Simple grid
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final safety check and clipping
    final_xy = best_result[:, :2]
    final_r = best_result[:, 2]
    
    # Clip radii to ensure they fit in walls
    for i in range(N):
        max_r = min(final_xy[i, 0], 1.0 - final_xy[i, 0], final_xy[i, 1], 1.0 - final_xy[i, 1])
        if final_r[i] > max_r:
            final_r[i] = max_r * 0.999
    
    # Verify final result
    if not _is_valid(final_xy, final_r):
         # If final check fails, return the grid fallback
         side = 5
         result = np.zeros((N, 3))
         for i in range(N):
             gx = i % side
             gy = i // side
             result[i, 0] = (gx + 0.5) / side
             result[i, 1] = (gy + 0.5) / side
             result[i, 2] = 0.4 / side
         return result

    return best_result

# EVOLVE-BLOCK-END
