import numpy as np
import time
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # Seconds, leaving margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _generate_initial_config():
    """
    Generate a high-quality hexagonal initial configuration.
    We try to fit a hexagonal lattice into the square.
    """
    # Hexagonal lattice parameters
    # Let's try a range of spacings to find the best fit for N=200
    best_centres = None
    best_r = -1
    
    # Approximate density of hex packing is ~0.9069
    # Area of square = 1. Area of N circles = N * pi * r^2.
    # 200 * pi * r^2 <= 0.9069 * 1  => r^2 <= 0.9069 / (200 * pi) ~ 0.00144 => r ~ 0.038
    # Current best known is ~0.0366.
    
    # We will generate a hex grid with a specific spacing 's' (distance between centers)
    # and then center and clip it to the square, ensuring we have exactly N points.
    
    # Strategy: Generate a large hex grid, then select the N points closest to the center?
    # No, that might leave large empty spaces at the edges.
    # Better: Generate a hex grid that fits "roughly" and then adjust.
    
    # Let's use the 20x10 rectangular grid distorted into a hex shape as a baseline,
    # but also try a true hexagonal lattice generation.
    
    # Method 1: Distorted Rectangular Grid (Current approach, reliable)
    w, h = 20, 10
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    
    # Scale to fit width 1 with some margin
    scale_x = 1.0 / (w - 1)
    scale_y = 1.0 / (h - 1)
    
    # Hex offset
    # In a hex lattice, vertical spacing is s * sqrt(3)/2
    # If we keep x-spacing = s, y-spacing should be s * sqrt(3)/2
    # Here we have fixed h=10, w=20.
    # Let's set s = scale_x. Then ideal y-spacing = scale_x * sqrt(3)/2.
    # Total height = (h-1) * scale_x * sqrt(3)/2.
    # If this height > 1, we scale down y.
    
    s = scale_x
    ideal_y_spacing = s * np.sqrt(3) / 2
    total_height = (h - 1) * ideal_y_spacing
    
    if total_height > 1.0:
        # Scale down y to fit
        scale_y = 1.0 / total_height
        y_spacing = 1.0 / (h - 1)
    else:
        y_spacing = ideal_y_spacing
        scale_y = 1.0 # Keep y as is, or center it
    
    # Generate points
    points = []
    for j in range(h):
        y = (j - (h-1)/2.0) * y_spacing
        # Offset for odd rows
        offset = 0
        if j % 2 != 0:
            offset = s / 2.0
            
        for i in range(w):
            x = (i - (w-1)/2.0) * s + offset
            points.append([x, y])
            
    grid = np.array(points)
    
    # Check bounds and clip/scale if necessary
    max_x = np.max(np.abs(grid[:, 0]))
    max_y = np.max(np.abs(grid[:, 1]))
    if max_x > 0.5 or max_y > 0.5:
        scale_down = 0.5 / max(max_x, max_y)
        grid *= scale_down
        
    r = _compute_radius(grid)
    if r > best_r:
        best_r = r
        best_centres = grid
        
    # Method 2: Try a few variations of hex spacing
    # We can vary the aspect ratio slightly
    for aspect in [0.9, 0.95, 1.0, 1.05, 1.1]:
        w_test, h_test = 20, 10
        s_test = 1.0 / (w_test - 1)
        y_spacing_test = s_test * np.sqrt(3) / 2 * aspect
        
        total_height_test = (h_test - 1) * y_spacing_test
        if total_height_test > 1.0:
            y_spacing_test = 1.0 / (h_test - 1)
            
        points = []
        for j in range(h_test):
            y = (j - (h_test-1)/2.0) * y_spacing_test
            offset = 0
            if j % 2 != 0:
                offset = s_test / 2.0
            for i in range(w_test):
                x = (i - (w_test-1)/2.0) * s_test + offset
                points.append([x, y])
                
        grid = np.array(points)
        max_x = np.max(np.abs(grid[:, 0]))
        max_y = np.max(np.abs(grid[:, 1]))
        if max_x > 0.5 or max_y > 0.5:
            scale_down = 0.5 / max(max_x, max_y)
            grid *= scale_down
            
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_centres = grid
            
    return best_centres

def _interior_point_energy_and_grad(centres, t, mu):
    """
    Interior-point formulation:
    Maximize t subject to:
      d_ij >= 2*t for all pairs (i,j)
      wall_dist_i >= t for all i
    
    Using log-barrier: minimize -t - mu * sum(log(constraints))
    where constraints are (d_ij - 2*t) and (wall_dist_i - t).
    
    We optimize over (centres, t) jointly.
    """
    centres = centres.reshape(N, 2)
    t = float(t)
    
    # Compute wall distances
    wx = 0.5 - np.abs(centres[:, 0])
    wy = 0.5 - np.abs(centres[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    # Wall constraints: wall_dist_i - t >= 0
    wall_constraints = wall_dists - t
    
    # Compute pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    
    # Pair constraints: d_ij - 2*t >= 0
    pair_constraints = dists / 2.0 - t
    # Only upper triangle
    idx_i, idx_j = np.triu_indices(N, k=1)
    pair_constraints = pair_constraints[idx_i, idx_j]
    
    # Check for constraint violations
    min_wall_c = np.min(wall_constraints)
    min_pair_c = np.min(pair_constraints)
    
    if min_wall_c < -1e-6 or min_pair_c < -1e-6:
        # Heavy penalty for violation
        penalty = 1e6 * (min(min_wall_c, 0)**2 + min(min_pair_c, 0)**2)
        return t + penalty, None  # Gradient not reliable when violated
    
    # Log barrier term
    # To avoid log of negative or zero, clip to small positive
    eps = 1e-12
    wall_c_clipped = np.maximum(wall_constraints, eps)
    pair_c_clipped = np.maximum(pair_constraints, eps)
    
    log_barrier = np.sum(np.log(wall_c_clipped)) + np.sum(np.log(pair_c_clipped))
    
    # Objective: maximize t + mu * log_barrier
    # So we minimize: -t - mu * log_barrier
    objective = -t - mu * log_barrier
    
    # Gradient computation
    grad_centres = np.zeros_like(centres)
    grad_t = 1.0  # d(-t)/dt = -1, so we negate: grad_t = 1 for minimization
    
    # Wall gradient contributions
    # d/dx_i [log(wall_dist_i - t)] = (1/(wall_dist_i - t)) * d(wall_dist_i)/dx_i
    for i in range(N):
        wc = wall_c_clipped[i]
        if wc < 1e-10:
            continue
        coeff = mu / wc
        if wx[i] < wy[i]:
            # wall_dist = wx = 0.5 - |x|
            # d(wx)/dx = -sign(x)
            grad_centres[i, 0] -= coeff * np.sign(centres[i, 0])
        elif wy[i] < wx[i]:
            # wall_dist = wy = 0.5 - |y|
            # d(wy)/dy = -sign(y)
            grad_centres[i, 1] -= coeff * np.sign(centres[i, 1])
        else:
            # Tie
            grad_centres[i, 0] -= 0.5 * coeff * np.sign(centres[i, 0])
            grad_centres[i, 1] -= 0.5 * coeff * np.sign(centres[i, 1])
        # Gradient of -mu*log(wall_c) w.r.t. t: +mu/(wall_c)
        grad_t += mu / wc
    
    # Pair gradient contributions
    # d/dx_i [log(d_ij/2 - t)] = (1/(d_ij/2 - t)) * (1/2) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j)/d_ij
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        pc = pair_c_clipped[k]
        if pc < 1e-10:
            continue
        coeff = mu / pc
        
        dx = centres[i, 0] - centres[j, 0]
        dy = centres[i, 1] - centres[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-12:
            continue
        
        # d(d_ij/2)/dx_i = dx/(2*dist)
        factor = coeff * 0.5 / dist
        
        grad_centres[i, 0] -= factor * dx
        grad_centres[i, 1] -= factor * dy
        grad_centres[j, 0] += factor * dx
        grad_centres[j, 1] += factor * dy
        
        # Gradient of -mu*log(pair_c) w.r.t. t: +mu/(pair_c)
        grad_t += mu / pc
    
    return objective, np.concatenate([grad_centres, [grad_t]])

def _interior_point_optimize(centres, t_init, mu_init, max_iters=200, time_budget=60.0):
    """
    Run interior-point optimization with adaptive mu.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_t = t_init
    best_r = _compute_radius(best_centres)
    
    x0 = np.concatenate([best_centres.ravel(), [best_t]])
    
    mu = mu_init
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Check if we're close to a feasible solution
        current_r = _compute_radius(best_centres)
        if current_r > best_r:
            best_r = current_r
        
        def obj_and_grad(x):
            if time.time() - start_time > time_budget - 2:
                return 1e10, np.zeros_like(x)
            c = x[:2*N].reshape(N, 2)
            t = x[2*N]
            c = np.clip(c, -0.4999, 0.4999)
            t = max(t, 1e-6)
            val, grad = _interior_point_energy_and_grad(c, t, mu)
            if grad is None:
                # Return a large value and zero gradient
                return val, np.zeros_like(x)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 50,
                    'maxfun': 100,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 20
                }
            )
            
            x_new = result.x
            new_centres = x_new[:2*N].reshape(N, 2)
            new_t = x_new[2*N]
            
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_t = max(new_t, 1e-6)
            
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
                best_t = new_t
                x0 = np.concatenate([best_centres.ravel(), [best_t]])
            
            # Decrease mu to tighten the barrier
            mu *= 0.5
            
            # If mu is very small, we're essentially solving the original problem
            if mu < 1e-6:
                break
                
        except Exception as e:
            break
    
    return best_centres, best_t, best_r

def _local_push_refine(centres, steps=50):
    """
    Refine the configuration by pushing apart the most constrained pairs.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    for step in range(steps):
        if time.time() > START_TIME + TIME_LIMIT - 5:
            break
            
        r = _compute_radius(best_centres)
        if r <= best_r:
            break
            
        # Find all pairs with distance close to 2*r
        # And points close to walls
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Identify active constraints
        # Threshold: within 1% of the minimum
        threshold = 2.0 * r * 1.01
        
        moved = False
        step_size = 0.001
        
        # Pairwise pushes
        # We want to move i and j apart.
        # Force direction: from j to i for point i, from i to j for point j.
        
        # To avoid conflicting moves, we can accumulate displacements
        displacements = np.zeros_like(best_centres)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < threshold:
                    # They are too close
                    dx = best_centres[i, 0] - best_centres[j, 0]
                    dy = best_centres[i, 1] - best_centres[j, 1]
                    norm = np.sqrt(dx*dx + dy*dy)
                    if norm < 1e-6:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        norm = 1.0
                    else:
                        norm = 1.0 # normalized direction
                    
                    # Push strength proportional to violation
                    violation = threshold - d
                    strength = step_size * (violation / (2*r)) # normalized
                    
                    displacements[i, 0] += strength * dx
                    displacements[i, 1] += strength * dy
                    displacements[j, 0] -= strength * dx
                    displacements[j, 1] -= strength * dy
                    moved = True
        
        # Wall pushes
        for i in range(N):
            wx = 0.5 - np.abs(best_centres[i, 0])
            wy = 0.5 - np.abs(best_centres[i, 1])
            
            if wx < r * 1.01:
                # Close to left/right wall
                sign_x = np.sign(best_centres[i, 0])
                if sign_x == 0: sign_x = 1
                displacements[i, 0] += step_size * sign_x
                moved = True
                
            if wy < r * 1.01:
                # Close to top/bottom wall
                sign_y = np.sign(best_centres[i, 1])
                if sign_y == 0: sign_y = 1
                displacements[i, 1] += step_size * sign_y
                moved = True
        
        if not moved:
            break
            
        # Apply displacements
        new_centres = best_centres + displacements
        # Clip to bounds to ensure validity
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_r = _compute_radius(new_centres)
        if new_r > best_r:
            best_centres = new_centres
            best_r = new_r
        else:
            # If it didn't improve, reduce step size or break
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    centres = _generate_initial_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    best_centres = centres.copy()
    best_r = current_r
    
    # 2. Interior Point Optimization
    if time_budget > 30:
        # Initial t estimate: use current radius
        t_init = best_r
        
        # Start with a moderate mu
        mu_init = 0.1
        
        # Allocate time for interior point
        ip_time = min(time_budget - 10, 80.0)
        
        best_centres, best_t, best_r_ip = _interior_point_optimize(
            best_centres, t_init, mu_init, max_iters=100, time_budget=ip_time
        )
        
        if best_r_ip > best_r:
            best_r = best_r_ip
    
    # 3. Local Push Refinement
    if time.time() < START_TIME + TIME_LIMIT - 5:
        best_centres, best_r = _local_push_refine(best_centres, steps=50)
        
    # 4. Final Validation
    final_r = _compute_radius(best_centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centres
