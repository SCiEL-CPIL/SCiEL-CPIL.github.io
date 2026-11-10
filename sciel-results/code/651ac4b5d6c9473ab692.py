import numpy as np
import time
import sys
from scipy.spatial.distance import cdist

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

def _fibonacci_lattice_init():
    """
    Generate initial configuration using a Fibonacci lattice (sunflower pattern).
    This provides a uniform distribution in the unit square.
    """
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Use square root of index for radial distribution to ensure uniform area density
    # r_i = sqrt(i / n)
    # theta_i = i * golden_angle
    
    # We map from polar (r, theta) in [0, 1] x [0, 2*pi) to square [-0.5, 0.5]^2
    # Actually, standard Fibonacci lattice maps to a circle.
    # For a square, we can use a different mapping or just generate in circle and scale/clip?
    # Clipping is bad.
    
    # Alternative: Use a "Fibonacci" like sequence for 2D grid?
    # Or just use a hex grid which is known to be good.
    # The directive says: "Initialize the packing using a Fibonacci lattice (sunflower pattern)"
    
    # Let's generate points in a unit circle using Fibonacci lattice, then map to square?
    # Or generate in square directly?
    
    # A common way to get uniform points in a square using low-discrepancy sequences (like Halton)
    # is better than Fibonacci (which is for circles).
    # But the directive specifically asks for Fibonacci.
    
    # Let's try a "Fibonacci" grid:
    # x_i = (i * phi) mod 1
    # y_i = (i * phi^2) mod 1
    # This is a 2D low-discrepancy sequence.
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n)
    x = (i * phi) % 1.0
    y = (i * phi * phi) % 1.0
    
    # Map [0, 1) to [-0.5, 0.5]
    centres = np.column_stack([x - 0.5, y - 0.5])
    
    return centres

def _hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    """
    best_centres = None
    best_r = -1
    
    # Try 20x10 grid with hex distortion
    w, h = 20, 10
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    
    # Scale to fit width 1
    scale = 1.0 / (w - 1)
    grid = np.array([(x * scale, y * scale) for y in ys for x in xs])
    
    # Apply hex offset to odd rows
    for i in range(h):
        if i % 2 != 0:
            grid[i*w:(i+1)*w, 0] += scale / 2.0
            
    # Check bounds and scale down if necessary
    max_x = np.max(np.abs(grid[:, 0]))
    max_y = np.max(np.abs(grid[:, 1]))
    if max_x > 0.5 or max_y > 0.5:
        scale_down = 0.5 / max(max_x, max_y)
        grid *= scale_down
        
    r = _compute_radius(grid)
    if r > best_r:
        best_r = r
        best_centres = grid
        
    # Also try pure square grid
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    scale = 1.0 / (w - 1)
    grid_sq = np.array([(x * scale, y * scale) for y in ys for x in xs])
    r_sq = _compute_radius(grid_sq)
    if r_sq > best_r:
        best_r = r_sq
        best_centres = grid_sq
        
    return best_centres

def _local_optimize(centres, time_limit):
    """
    Optimize the positions to maximize the minimum radius using a repulsive force-based simulation.
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    step_size = 0.01
    min_step_size = 1e-6
    cooling = 1.0
    min_cooling = 0.01
    
    # We will use a force-based approach.
    # Each point feels a repulsive force from other points and walls.
    # The force is inversely proportional to distance (or squared).
    # We move points in the direction of the net force.
    
    # To improve efficiency, we can use a grid-based neighbor search or just cdist for N=200.
    # cdist is O(N^2) which is 40000 operations, very fast.
    
    while time.time() < deadline and step_size > min_step_size:
        # Calculate forces
        forces = np.zeros_like(best_centres)
        
        # Pairwise repulsion
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: 1 / d^2 (Coulomb-like) or 1/d
        # Using 1/d^2 creates stronger repulsion at close range.
        # We only care about the minimum distance, so focusing on close pairs is good.
        
        # To avoid division by zero and numerical issues, add a small epsilon
        eps = 1e-6
        inv_d2 = np.zeros_like(dists)
        mask = dists < 0.2 # Only consider close neighbors to save computation?
        # Actually, for N=200, we can compute all.
        
        # We want to maximize the minimum distance.
        # A common potential is U = 1/d. Force = 1/d^2.
        # Let's use a soft potential: U = 1/d^2, Force = 2/d^3.
        # Or simply move points apart based on the direction to the nearest neighbor.
        
        # Let's stick to the "push" strategy but with a more global view.
        # Calculate the "pressure" on each point from all others.
        
        # For each pair (i, j), if dist < 2*r_target, apply repulsive force.
        # What is r_target? We can use the current best_r or a slightly larger value.
        # Let's use a fixed target radius that we try to achieve.
        
        # Alternative: Gradient ascent on the soft-min of distances.
        # Let f(x) = sum_i sum_{j>i} exp(-lambda * d_ij) + sum_i exp(-lambda * w_i)
        # Minimizing f is equivalent to maximizing the minimum distance (approx).
        
        # Let's implement a simple gradient-based step.
        # Objective: Maximize r = min(min_pair_dist/2, min_wall_dist)
        # This is non-smooth.
        
        # Use a surrogate: Maximize sum of exp(-k * d_ij) for close pairs, and exp(-k * w_i) for walls.
        # The gradient of -sum exp(-k d_ij) w.r.t x_i is sum_j k * exp(-k d_ij) * (x_i - x_j) / d_ij
        
        k = 20.0 # Sharpness of the soft-min
        
        # Compute distances again
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Weights for each pair
        weights = np.exp(-k * dists)
        
        # Gradient of the objective (to be maximized)
        # d/dx_i sum_{j != i} exp(-k * |x_i - x_j|) = sum_{j != i} (-k) * exp(-k * d_ij) * (x_i - x_j) / d_ij
        # So we want to move in the direction of the negative gradient of the sum of exp(-k d)
        # Which is the direction of the gradient of the sum of -exp(-k d)
        
        grad = np.zeros_like(best_centres)
        
        # Vectorized computation of gradient
        # grad_i = sum_j k * exp(-k d_ij) * (x_i - x_j) / d_ij
        
        # Let's compute this carefully
        # diff = best_centres[:, np.newaxis, :] - best_centres[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # We already have dists.
        
        # We need (x_i - x_j) / d_ij
        # Let's compute the unit vectors
        diff = best_centres[:, np.newaxis, :] - best_centres[np.newaxis, :, :]
        # dists is the norm
        # Avoid division by zero
        safe_dists = np.where(dists < 1e-6, 1.0, dists)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Weighted sum
        # weights is (N, N)
        # We want sum_j weights[i, j] * unit_vecs[i, j]
        # Note: unit_vecs[i, j] points from j to i.
        # Force on i from j is proportional to (x_i - x_j) / d_ij
        # So grad_i = sum_j k * weights[i, j] * unit_vecs[i, j]
        
        grad = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1) * k
        
        # Wall forces
        # We want to push points away from walls if they are too close.
        # Wall distance w_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # Surrogate: sum_i exp(-k * w_i)
        # Gradient of -exp(-k w_i) w.r.t x_i is -(-k) * exp(-k w_i) * d(w_i)/dx_i = k * exp(-k w_i) * d(w_i)/dx_i
        
        # d(w_i)/dx_i:
        # If x_i > 0, w_i = min(0.5 - x_i, ...). If 0.5 - x_i is the min, d(w_i)/dx_i = -1.
        # If x_i < 0, w_i = min(0.5 + x_i, ...). If 0.5 + x_i is the min, d(w_i)/dx_i = 1.
        
        wall_weights = np.zeros(N)
        wall_grad = np.zeros_like(best_centres)
        
        dx = best_centres[:, 0]
        dy = best_centres[:, 1]
        
        dist_left = 0.5 + dx
        dist_right = 0.5 - dx
        dist_down = 0.5 + dy
        dist_up = 0.5 - dy
        
        min_wall_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_down, dist_up))
        
        # Only apply wall force if close to wall
        # If min_wall_dist is small, exp(-k * min_wall_dist) is large.
        wall_weights = np.exp(-k * min_wall_dist)
        
        # Determine direction
        # For x:
        # If dist_left < dist_right and dist_left < min(dist_down, dist_up), then wall is left, push right (+1)
        # If dist_right < dist_left and dist_right < min(dist_down, dist_up), then wall is right, push left (-1)
        
        # We can use a soft selection or just hard selection based on which is min.
        # For simplicity, let's use the gradient of the min function approximated by the smallest distance.
        
        # x-component of wall gradient
        # If left wall is closest: d(w)/dx = -1 => grad_x += k * exp(-k w) * (-1) ?
        # Wait, we are maximizing r.
        # Objective term: -exp(-k w). Gradient w.r.t x: k * exp(-k w) * dw/dx.
        # If left wall (x = -0.5) is closest, w = 0.5 + x. dw/dx = 1.
        # So we want to move in +x direction.
        # Formula: grad_x += k * exp(-k w) * 1.
        
        # If right wall (x = 0.5) is closest, w = 0.5 - x. dw/dx = -1.
        # Formula: grad_x += k * exp(-k w) * (-1).
        
        # Let's compute which wall is closest for each point.
        # We can do this by comparing distances.
        
        # For x:
        # If dist_left <= dist_right:
        #    If dist_left <= dist_down and dist_left <= dist_up:
        #        grad_x += k * exp(-k dist_left) * 1
        #    else:
        #        # Wall is not x, but maybe y?
        #        pass
        # Else:
        #    If dist_right <= dist_down and dist_right <= dist_up:
        #        grad_x += k * exp(-k dist_right) * (-1)
        
        # This is getting complicated to vectorize cleanly with mins.
        # Let's just compute the gradient of the soft-min of the 4 wall distances.
        # w_i = min(dL, dR, dD, dU)
        # Soft min: sm = -1/k * log(sum(exp(-k * d)))
        # Gradient of -sm = 1/k * d/dx log(sum(exp(-k d))) = 1/k * (sum(-k exp(-k d) d/dx) / sum(exp(-k d)))
        # = - sum(exp(-k d) d/dx) / sum(exp(-k d))
        
        # Let's use this soft-min formulation for walls too.
        # It's smoother.
        
        dL = 0.5 + dx
        dR = 0.5 - dx
        dD = 0.5 + dy
        dU = 0.5 - dy
        
        # Stack wall distances
        wall_dists = np.stack([dL, dR, dD, dU], axis=1) # (N, 4)
        
        # Soft min
        # sm = -1/k * logsumexp(-k * wall_dists)
        # logsumexp(x) = log(sum(exp(x)))
        
        # Gradient of -sm w.r.t x_i
        # d(-sm)/dx = - d/dx (-1/k log(sum(exp(-k d))))
        # = 1/k * (1/sum(exp(-k d))) * sum(d/dx exp(-k d))
        # = 1/k * (1/sum(exp(-k d))) * sum(exp(-k d) * (-k) * d(d)/dx)
        # = - sum(exp(-k d) * d(d)/dx) / sum(exp(-k d))
        
        # d(dL)/dx = 1, d(dR)/dx = -1, d(dD)/dx = 0, d(dU)/dx = 0
        # d(dL)/dy = 0, d(dR)/dy = 0, d(dD)/dy = 1, d(dU)/dy = -1
        
        # Let's compute the weighted average of the gradients of the individual distances.
        # weights_w = exp(-k * wall_dists) / sum(exp(-k * wall_dists))
        
        exp_walls = np.exp(-k * wall_dists)
        sum_exp_walls = np.sum(exp_walls, axis=1, keepdims=True)
        weights_w = exp_walls / sum_exp_walls
        
        # Gradient of wall term for x
        # grad_x_wall = - (weights_w[:, 0] * 1 + weights_w[:, 1] * (-1) + 0 + 0)
        # = - (weights_w[:, 0] - weights_w[:, 1])
        
        # Gradient of wall term for y
        # grad_y_wall = - (0 + 0 + weights_w[:, 2] * 1 + weights_w[:, 3] * (-1))
        # = - (weights_w[:, 2] - weights_w[:, 3])
        
        wall_grad[:, 0] = -(weights_w[:, 0] - weights_w[:, 1])
        wall_grad[:, 1] = -(weights_w[:, 2] - weights_w[:, 3])
        
        # Combine gradients
        # Total gradient = grad_pair + wall_grad
        # We want to move in the direction of the gradient (ascent).
        
        total_grad = grad + wall_grad
        
        # Normalize gradient to prevent large steps
        grad_norms = np.linalg.norm(total_grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms < 1e-6, 1.0, grad_norms)
        normalized_grad = total_grad / grad_norms
        
        # Update positions
        new_centres = best_centres + step_size * cooling * normalized_grad
        
        # Clip to bounds
        new_centres = np.clip(new_centres, -0.5 + 1e-6, 0.5 - 1e-6)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            cooling *= 0.99 # Slow cooling
        else:
            # If no improvement, try a smaller step or random kick
            step_size *= 0.95
            if step_size < 1e-4:
                # Random kick
                kick_idx = np.random.randint(N)
                kick_x = np.random.uniform(-step_size*10, step_size*10)
                kick_y = np.random.uniform(-step_size*10, step_size*10)
                new_centres = best_centres.copy()
                new_centres[kick_idx] += [kick_x, kick_y]
                new_centres[kick_idx] = np.clip(new_centres[kick_idx], -0.5, 0.5)
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    step_size = 0.01
                    cooling = 1.0
                else:
                    step_size *= 0.8
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    # Try both Fibonacci and Hexagonal, pick the best
    centres_fib = _fibonacci_lattice_init()
    r_fib = _compute_radius(centres_fib)
    
    centres_hex = _hexagonal_init()
    r_hex = _compute_radius(centres_hex)
    
    if r_fib > r_hex:
        centres = centres_fib
        current_r = r_fib
    else:
        centres = centres_hex
        current_r = r_hex
        
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # 2. Local Optimization
    if time_budget > 10:
        centres, current_r = _local_optimize(centres, time_budget - 5)
        
    # 3. Final Validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
