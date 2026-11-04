"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _compute_radii(xy, max_iter=50, tol=1e-6):
    """
    Given centers xy, compute the maximum feasible radii for each circle.
    Uses an iterative relaxation: r_i = min(
        min(walls),
        min_j (dist_ij - r_j)
    )
    This is a fixed-point iteration that converges to the maximal radii 
    for the given topology (which circle touches which).
    """
    n = len(xy)
    r = np.full(n, 0.1) # Initial guess
    
    # Precompute wall limits
    wall_limits = np.minimum(np.minimum(xy[:, 0], 1.0 - xy[:, 0]), 
                             np.minimum(xy[:, 1], 1.0 - xy[:, 1]))
    
    for _ in range(max_iter):
        new_r = wall_limits.copy()
        
        # For each pair, update constraint: r_i + r_j <= dist_ij
        # => r_i <= dist_ij - r_j
        # We update in place or use a copy to ensure stability.
        # Using Gauss-Seidel style update for faster convergence.
        for i in range(n):
            current_max = wall_limits[i]
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # If dist is very small, this constraint is tight and bad
                limit = dist - r[j]
                if limit < current_max:
                    current_max = limit
            new_r[i] = max(0.001, current_max)
        
        # Check convergence
        if np.max(np.abs(new_r - r)) < tol:
            r = new_r
            break
        r = new_r
        
    return r

def _objective_and_grad(xy, beta=10.0):
    """
    Objective: -sum(radii) + penalty for overlaps (if any, though radii are derived, 
    numerical noise might cause slight issues, but mainly we just maximize sum of radii).
    Actually, since radii are derived from xy, the function is just -sum(_compute_radii(xy)).
    However, _compute_radii is non-smooth (min operations). 
    We use a soft-min approximation for gradients if needed, but 
    scipy's L-BFGS-B can handle non-smooth functions poorly.
    
    Better approach: 
    The problem of maximizing sum of radii given centers is equivalent to 
    finding the "Voronoi-like" packing.
    Since _compute_radii is piecewise linear (in terms of distances), 
    we can use subgradient or just finite differences if gradients are tricky.
    But let's try to implement a smooth approximation or just use 
    a gradient-free method like Powell or Nelder-Mead on the centers, 
    or L-BFGS-B with a smoothed objective.
    
    Let's use a smooth penalty approach for the objective in optimization:
    Maximize sum(r_i) subject to r_i + r_j <= dist_ij and r_i <= wall_i.
    This is a Linear Program if r_i and dist_ij are fixed, but dist_ij depends on xy.
    
    Alternative: 
    We optimize xy. The value is V(xy) = sum(r_i(xy)).
    We can approximate the gradient of r_i with respect to xy_k.
    r_i is the min of several terms. The active term determines the gradient.
    If r_i is limited by wall, grad is 0 w.r.t other centers, but w.r.t its own center it's 1 or -1.
    If r_i is limited by circle j, then r_i = dist_ij - r_j.
    This creates a system of equations.
    
    For simplicity and robustness, we will use **Nelder-Mead** or **Powell** 
    on the centers, as they don't require gradients and handle the 
    non-smoothness of the min() function in radius calculation reasonably well 
    for 52 dimensions (26 circles * 2 coords).
    However, 52 dimensions is high for Nelder-Mead.
    
    Let's stick to L-BFGS-B but use a **smoothed** radius calculation.
    Instead of hard min, use a soft-min (log-sum-exp with negative sign).
    """
    n = len(xy)
    r = np.full(n, 0.1)
    
    # Soft min iteration
    # r_i = -1/beta * log( sum( exp(-beta * limit_k) ) )
    # where limits are wall_i and dist_ij - r_j
    
    for _ in range(10): # Fewer iterations for speed in inner loop
        new_r = np.zeros(n)
        for i in range(n):
            # Wall limits
            limits = [
                xy[i, 0],
                1.0 - xy[i, 0],
                xy[i, 1],
                1.0 - xy[i, 1]
            ]
            # Circle limits
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                limits.append(dist - r[j])
            
            limits = np.array(limits)
            # Soft min: approximates min(limits)
            # To avoid overflow, subtract max
            m = np.max(limits)
            soft_min = m - (1.0/beta) * np.log(np.sum(np.exp(-beta * (limits - m))))
            new_r[i] = max(0.001, soft_min)
        r = new_r
        
    total_r = np.sum(r)
    return -total_r, r

def _smooth_objective(v, beta=10.0):
    xy = v.reshape(-1, 2)
    val, _ = _objective_and_grad(xy, beta)
    return val

def _smooth_grad(v, beta=10.0):
    # Finite difference gradient for robustness against non-smoothness
    eps = 1e-5
    grad = np.zeros_like(v)
    f0 = _smooth_objective(v, beta)
    for i in range(len(v)):
        v_plus = v.copy()
        v_plus[i] += eps
        f_plus = _smooth_objective(v_plus, beta)
        grad[i] = (f_plus - f0) / eps
    return grad

def _optimize_centers(xy_init, deadline, beta_start=5.0, beta_end=50.0, num_steps=5):
    """
    Optimize centers using L-BFGS-B with increasing beta (sharpening the soft-min).
    """
    current_xy = xy_init.copy()
    v = current_xy.flatten()
    
    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_val = float('inf')
    best_xy = current_xy.copy()
    
    for step in range(num_steps):
        if time.time() > deadline:
            break
        # Interpolate beta
        beta = beta_start * (beta_end / beta_start) ** (step / max(1, num_steps - 1))
        
        try:
            res = minimize(
                _smooth_objective,
                v,
                jac=_smooth_grad,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_xy = res.x.reshape(-1, 2)
                v = res.x
        except Exception:
            pass
            
    return best_xy

def _local_refine(xy, deadline):
    """
    Use a more precise radius calculation and local search on centers 
    to squeeze out extra value.
    """
    current_xy = xy.copy()
    current_r = _compute_radii(current_xy)
    best_sum = np.sum(current_r)
    
    # Simple hill climbing on centers with small perturbations
    step_size = 0.01
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_xy = current_xy.copy()
                    new_xy[i, 0] += dx
                    new_xy[i, 1] += dy
                    # Keep in bounds
                    new_xy[i, 0] = np.clip(new_xy[i, 0], 0.0, 1.0)
                    new_xy[i, 1] = np.clip(new_xy[i, 1], 0.0, 1.0)
                    
                    new_r = _compute_radii(new_xy, max_iter=20)
                    new_sum = np.sum(new_r)
                    if new_sum > best_sum + 1e-6:
                        best_sum = new_sum
                        current_xy = new_xy
                        current_r = new_r
                        improved = True
        if not improved:
            step_size *= 0.8
            if step_size < 0.001:
                break
                
    return current_xy, current_r

def _generate_initial(xy_type, seed=0):
    rng = np.random.RandomState(seed)
    if xy_type == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    positions.append([x, y])
        return np.array(positions[:N])
    elif xy_type == 'random':
        return rng.uniform(0.1, 0.9, size=(N, 2))
    else:
        # Hexagonal-ish
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    if i % 2 == 1:
                        y += 0.25 / side
                    positions.append([x, y])
        return np.array(positions[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_xy = None
    best_r = None
    best_sum = 0.0
    
    initializations = ['grid', 'random', 'hex', 'random', 'grid']
    seeds = [0, 1, 2, 3, 4]
    
    for idx, (init_type, seed) in enumerate(zip(initializations, seeds)):
        if time.time() > deadline - 20:
            break
            
        xy_init = _generate_initial(init_type, seed)
        
        # Phase 1: Optimize centers with soft-min radii
        optimized_xy = _optimize_centers(xy_init, deadline - 10, beta_start=2.0, beta_end=20.0, num_steps=3)
        
        # Phase 2: Local refinement with exact radii
        refined_xy, refined_r = _local_refine(optimized_xy, deadline - 5)
        
        current_sum = np.sum(refined_r)
        
        # Validate
        valid = True
        for i in range(N):
            x, y, r = refined_xy[i, 0], refined_xy[i, 1], refined_r[i]
            if r < -1e-9 or x - r < -1e-9 or 1.0 - x - r < -1e-9 or y - r < -1e-9 or 1.0 - y - r < -1e-9:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = refined_xy[i, 0] - refined_xy[j, 0]
                    dy = refined_xy[i, 1] - refined_xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < refined_r[i] + refined_r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid and current_sum > best_sum:
            best_sum = current_sum
            best_xy = refined_xy
            best_r = refined_r

    # Fallback if optimization failed
    if best_xy is None:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final construction
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_xy[i, 0]
        result[i, 1] = best_xy[i, 1]
        result[i, 2] = best_r[i]
        
    return result
# EVOLVE-BLOCK-END
