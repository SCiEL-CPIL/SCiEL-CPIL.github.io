"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_max_radii(centers, max_iters=200, tol=1e-12):
    """
    Compute the maximum possible radii for a given set of centers
    such that circles are disjoint and inside the unit square.
    Uses iterative relaxation (Gauss-Seidel style).
    """
    # Wall distances
    wall_dist = np.min(np.minimum(centers, 1.0 - centers), axis=1)
    
    # Initialize radii to wall distances
    r = wall_dist.copy()
    
    # Precompute distance matrix
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    
    # Iterative relaxation
    for _ in range(max_iters):
        # constraints[i, j] = d[i, j] - r[j]
        # We want r[i] <= constraints[i, j] for all j != i
        constraints = d - r[np.newaxis, :]
        
        # Set diagonal to inf
        np.fill_diagonal(constraints, np.inf)
        
        # Min constraint for each i
        min_constraint = np.min(constraints, axis=1)
        
        # New radius is min(wall_dist, min_constraint)
        new_r = np.minimum(wall_dist, min_constraint)
        
        # Check convergence
        if np.max(np.abs(new_r - r)) < tol:
            r = new_r
            break
        r = new_r
        
    # Ensure radii are non-negative
    r = np.maximum(r, 0.0)
    
    return r

def _soft_min(a, beta=100.0):
    """
    Soft minimum of an array using log-sum-exp trick.
    Returns a smooth approximation to min(a).
    """
    # For stability, subtract max
    a_max = np.max(a)
    # If all are very negative, this might underflow, but we add a_max back
    # log(sum(exp(a_i - a_max))) + a_max = -logsumexp(-a_i)
    # min(a) ≈ -logsumexp(-a) / 1? No.
    # min(a) ≈ - (1/beta) * log(sum(exp(-beta * a_i)))
    # Let's use: min(a) ≈ - (1/beta) * log(sum(exp(-beta * a)))
    # To avoid overflow in exp(-beta * a), subtract max(-beta * a) = -beta * min(a)
    # But we don't know min(a) yet.
    # Standard stable softmin:
    # s = min(a)
    # approx = s - (1/beta) * log(sum(exp(-beta * (a_i - s))))
    # Since we want to avoid computing s exactly first, we can use:
    # approx = - (1/beta) * logsumexp(-beta * a)
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    # z = -beta * a
    # max(z) = -beta * min(a)
    # So we need min(a) for stability?
    # Actually, just using np.logaddexp.reduce is stable enough for moderate beta.
    # Or:
    # - (1/beta) * log( sum( exp(-beta * a) ) )
    # Let's implement stable version:
    # m = np.min(a)
    # return m - (1/beta) * np.log(np.sum(np.exp(-beta * (a - m))))
    
    m = np.min(a)
    # If m is -inf, return -inf
    if not np.isfinite(m):
        return m
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (a - m))))

def _objective_smooth(centers, beta=50.0):
    """
    Smooth objective function to minimize: Negative sum of soft-min radii.
    This is differentiable, enabling gradient-based optimization.
    centers: flat array of length 2*N
    """
    pos = centers.reshape(N, 2)
    
    # Wall distances (vectorized)
    # dist to left: x, right: 1-x, bottom: y, top: 1-y
    wall_dists = np.stack([
        pos[:, 0], 
        1.0 - pos[:, 0], 
        pos[:, 1], 
        1.0 - pos[:, 1]
    ], axis=1) # shape (N, 4)
    
    # Soft min over walls for each circle
    # We want min over the 4 walls
    r_wall = np.zeros(N)
    for i in range(N):
        r_wall[i] = _soft_min(wall_dists[i], beta)
        
    # Pairwise distances
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    
    # For each circle i, we need min over j!=i of (d[i,j] - r[j])
    # This is tricky because r[j] depends on pos.
    # In the smooth relaxation, we approximate the fixed point.
    # Let's assume r is the fixed point of the relaxation.
    # For a smooth objective, we can just compute the radii using the smooth relaxation
    # and sum them.
    
    # Initialize r
    r = r_wall.copy()
    
    # Iterate a few times to get a smooth fixed point
    # We can do a few Gauss-Seidel steps, but they are not differentiable due to min.
    # Instead, we can use the soft-min directly in the constraint.
    # r_i <= min_j (d_ij - r_j)
    # r_i <= min( r_wall_i, min_j (d_ij - r_j) )
    
    # Let's just use the standard relaxation but with soft-min?
    # No, the relaxation is the definition of the max radii.
    # The function f(pos) = sum(r_max(pos)) is concave and piecewise linear.
    # Its gradient is discontinuous at the kinks.
    # Using a soft-min surrogate for the "min" operation in the relaxation
    # gives a smooth approximation to the radii.
    
    # Let's try a simple approach:
    # Run the standard relaxation for a few steps, but replace the hard min with soft min?
    # Or just use the hard min and rely on SLSQP's finite difference gradients?
    # SLSQP with finite differences on a non-smooth function can be unstable.
    
    # Alternative: Use the "barrier" method or just penalty.
    # Let's stick to the standard relaxation for the value, but for optimization
    # use a penalty on overlaps and boundaries, and maximize sum of radii
    # where radii are computed by the relaxation.
    # Actually, the relaxation IS the solution to the dual problem.
    # The value is concave.
    
    # Let's just return the negative sum of the exact max radii.
    # The "smooth" part comes from using SLSQP or similar which can handle
    # some non-smoothness, or we can add a small penalty for overlaps
    # to guide the optimizer away from the kinks?
    
    # For now, let's just use the exact computation. It's fast enough.
    # The "smooth" label is for the future, or if we implement a differentiable relaxation.
    # But for now, let's just compute the exact radii.
    
    # Wait, the prompt suggests using a smooth surrogate.
    # Let's implement a differentiable approximation of the radii.
    # r_i = min( w_i, min_{j!=i} (d_ij - r_j) )
    # This is a system of equations.
    # We can approximate min(a, b) with soft_min(a, b, beta).
    
    r = r_wall.copy()
    for _ in range(5): # Few iterations
        # constraints[i, j] = d[i, j] - r[j]
        constraints = d - r[np.newaxis, :]
        np.fill_diagonal(constraints, np.inf)
        
        # For each i, min over j of constraints[i, j]
        # Use soft min
        min_constraints = np.zeros(N)
        for i in range(N):
            min_constraints[i] = _soft_min(constraints[i], beta)
            
        # New r_i = soft_min(wall_dist_i, min_constraints_i)
        # wall_dist_i is a scalar, min_constraints_i is a scalar
        # We need soft_min of two numbers
        # soft_min(a, b) = - (1/beta) * log(exp(-beta*a) + exp(-beta*b))
        # Or use the array version with [a, b]
        r_new = np.zeros(N)
        for i in range(N):
            r_new[i] = _soft_min(np.array([r_wall[i], min_constraints[i]]), beta)
            
        r = r_new
        
    return -np.sum(r)

def _initial_lattice():
    """Generate a hexagonal lattice-like initial state."""
    pos = np.zeros((N, 2))
    
    # Try to fit a hexagonal grid
    # For 26 circles, a 5x5 grid is 25, plus 1?
    # Or 6 rows of 4-5?
    # Let's just place them in a grid and let the optimizer fix it.
    
    # 5 columns, 5 rows = 25 circles.
    # 1 extra circle.
    
    # Grid spacing
    # If we use r=0.1, diameter 0.2.
    # 5 * 0.2 = 1.0. Fits exactly.
    # So centers at 0.1, 0.3, 0.5, 0.7, 0.9
    
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ys = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    k = 0
    for y in ys:
        for x in xs:
            if k < N:
                pos[k] = [x, y]
                k += 1
            else:
                break
        if k >= N:
            break
            
    # If we have more than 25, add the 26th somewhere
    if k < N:
        # Add at center of a cell?
        pos[k] = [0.2, 0.2] # Just a guess
        k += 1
        
    return pos

def _initial_random(seed):
    """Generate a random initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    return pos

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm on centers.
    """
    pos = np.zeros((N, 2))
    pos[0] = [0.5, 0.5]
    
    for i in range(1, N):
        best_x, best_y = 0.5, 0.5
        best_score = -np.inf
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample random points
        for _ in range(500):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Temporarily add this point
            temp_pos = pos[:i+1].copy()
            temp_pos[i] = [x, y]
            
            # Compute radii for this temp set
            radii = _compute_max_radii(temp_pos, max_iters=10)
            
            # Score is sum of radii
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_x, best_y = x, y
                
        pos[i] = [best_x, best_y]
        
    return pos

def _simulated_annealing(pos_init, time_limit=10.0, seed=42):
    """
    Simulated Annealing to optimize centers.
    """
    rng = np.random.RandomState(seed)
    pos = pos_init.copy()
    
    # Initial radii
    radii = _compute_max_radii(pos)
    current_score = np.sum(radii)
    best_score = current_score
    best_pos = pos.copy()
    
    T = 0.05
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.05
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        
        # Propose new position
        noise = rng.normal(0, step_size, (N, 2))
        new_pos = pos + noise
        
        # Clamp to [0, 1]
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # Compute new radii and score
        new_radii = _compute_max_radii(new_pos)
        new_score = np.sum(new_radii)
        
        # Delta E (we want to maximize score, so minimize -score)
        delta = new_score - current_score
        
        if delta > 0 or rng.uniform() < np.exp(delta / T):
            pos = new_pos
            current_score = new_score
            radii = new_radii
            
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
        
        # Cool down
        T *= cooling
        
    return best_pos, best_score

def _nelder_mead_polish(pos_init, time_limit=5.0):
    """
    Nelder-Mead polish on the centers.
    """
    x0 = pos_init.flatten()
    
    def obj_wrapper(x):
        pos = x.reshape(N, 2)
        if np.any(pos < 0) or np.any(pos > 1):
            return 1e6
        return _objective_smooth(x, beta=50.0)
        
    try:
        res = minimize(obj_wrapper, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        pos_opt = res.x.reshape(N, 2)
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        return pos_opt
        
    except Exception:
        return pos_init

def _slsqp_polish(pos_init, time_limit=10.0):
    """
    SLSQP polish using the smooth objective.
    """
    x0 = pos_init.flatten()
    
    # Bounds
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    def obj(x):
        return _objective_smooth(x, beta=100.0)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        
        pos_opt = res.x.reshape(N, 2)
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        return pos_opt
        
    except Exception:
        return pos_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Generate initial guesses
    inits = []
    inits.append(_initial_lattice())
    inits.append(_initial_best_fit())
    inits.append(_initial_random(123))
    inits.append(_initial_random(456))
    
    best_pos = inits[0]
    best_score = -np.inf
    
    # Run SA on each initial guess
    time_per_init = 20.0
    
    for i, pos_init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Run SA
        pos_out, score = _simulated_annealing(pos_init, time_limit=time_per_init, seed=100+i)
        
        if score > best_score:
            best_score = score
            best_pos = pos_out
            
    # Polish the best result with Nelder-Mead
    if time.time() < deadline - 10.0:
        time_left = deadline - time.time() - 2.0
        if time_left > 2.0:
            polished_pos = _nelder_mead_polish(best_pos, time_limit=time_left)
            radii = _compute_max_radii(polished_pos)
            polish_score = np.sum(radii)
            
            if polish_score > best_score:
                best_score = polish_score
                best_pos = polished_pos
                
    # Polish with SLSQP
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 2.0
        if time_left > 2.0:
            polished_pos = _slsqp_polish(best_pos, time_limit=time_left)
            radii = _compute_max_radii(polished_pos)
            polish_score = np.sum(radii)
            
            if polish_score > best_score:
                best_score = polish_score
                best_pos = polished_pos
                
    # Final computation of radii
    final_radii = _compute_max_radii(best_pos)
    
    # Ensure strict validity
    for i in range(N):
        r = final_radii[i]
        x, y = best_pos[i]
        max_r_wall = min(x, 1.0-x, y, 1.0-y)
        if r > max_r_wall:
            final_radii[i] = max_r_wall * 0.9999
            
    # Check overlaps one last time and shrink if necessary
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = best_pos[i, 0] - best_pos[j, 0]
                dy = best_pos[i, 1] - best_pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = final_radii[i] + final_radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        final_radii[i] *= k
                        final_radii[j] *= k
                        valid = False
        if valid:
            break
            
    result = np.column_stack([best_pos, final_radii])
    return result

# EVOLVE-BLOCK-END
