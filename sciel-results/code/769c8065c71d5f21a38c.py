"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance using cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _fibonacci_lattice():
    """Create a Fibonacci lattice (sunflower pattern) initialization."""
    # Generate N points using the golden angle
    n = N
    theta = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    i = np.arange(n)
    r = np.sqrt(i / n) * 0.5  # scale to fit in radius 0.5
    angles = i * theta
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    centres = np.column_stack((x, y))
    # Clip to ensure inside square
    centres = np.clip(centres, -0.49, 0.49)
    return centres


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                
                # Create the grid
                centres = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres.copy()
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _energy_and_gradient(centres_flat):
    """
    Compute energy and gradient for L-BFGS-B.
    Energy is sum of inverse distances (pairwise + walls).
    Minimizing this maximizes the minimum distance.
    """
    centres = centres_flat.reshape(-1, 2)
    energy = 0.0
    grad = np.zeros_like(centres)
    
    # Pairwise distances
    # To avoid O(N^2) in Python loop, we use cdist but need gradient.
    # For N=200, O(N^2) is 40,000 pairs. Python loop might be slow.
    # Let's try a vectorized approach for energy, and approximate gradient or use analytic.
    
    # Vectorized Energy:
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero if any coincident points (shouldn't happen with random init)
    dists = np.maximum(dists, 1e-6)
    
    # Sum of inverse distances
    pair_energy = np.sum(1.0 / dists)
    
    # Wall distances
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    wall_dists = np.maximum(wall_dists, 1e-6)
    wall_energy = np.sum(1.0 / wall_dists)
    
    total_energy = pair_energy + wall_energy
    
    # Gradient Calculation
    # d/dx_i (1/d_ij) = -1/d_ij^2 * d/dx_i (d_ij)
    # d_ij = ||x_i - x_j||
    # d/dx_i (d_ij) = (x_i - x_j) / d_ij
    # So d/dx_i (1/d_ij) = - (x_i - x_j) / d_ij^3
    
    # We can compute this vectorized for the full matrix, then sum over j for each i.
    # Let D be the distance matrix.
    # Grad_i = - sum_{j!=i} (x_i - x_j) / D_ij^3
    
    # Create a matrix of (x_i - x_j)
    # x_i[:, None, :] - x_j[None, :, :]
    diff = centres[:, None, :] - centres[None, :, :] # Shape (N, N, 2)
    
    # D_ij^3
    d_cubed = dists ** 3
    np.fill_diagonal(d_cubed, 1.0) # Avoid div by 0, we will zero out diagonal later
    
    # Contribution to gradient from each pair (i, j) is - (x_i - x_j) / d^3
    # Sum over j for each i.
    # Note: The term for (i, j) in the sum for i is - (x_i - x_j)/d^3
    # The term for (j, i) in the sum for j is - (x_j - x_i)/d^3 = (x_i - x_j)/d^3
    # So grad_i = sum_{j!=i} - (x_i - x_j) / d_ij^3
    
    # We can use matrix multiplication or just sum along axis 1
    # weights = 1 / d^3
    weights = 1.0 / d_cubed
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = - sum_j (x_i - x_j) * weights[i, j]
    # This is equivalent to:
    # grad_i = - (x_i * sum_j weights[i,j] - sum_j x_j * weights[i,j])
    
    sum_weights = np.sum(weights, axis=1, keepdims=True) # (N, 1)
    weighted_x_sum = weights @ centres # (N, 2)
    
    pair_grad = - (centres * sum_weights - weighted_x_sum)
    
    # Wall Gradient
    # d/dx_i (1/d_wall) = -1/d_wall^2 * d/dx_i (d_wall)
    # d_wall is min of 4 distances. The gradient is non-zero only for the closest wall.
    # We need to be careful with the min.
    # Let's compute gradients for all 4 walls and select the one corresponding to the min distance.
    
    # Distances to walls:
    d_left = 0.5 - centres[:, 0]
    d_right = 0.5 + centres[:, 0]
    d_bottom = 0.5 - centres[:, 1]
    d_top = 0.5 + centres[:, 1]
    
    # Gradients of these distances w.r.t (x, y):
    # d_left: (-1, 0)
    # d_right: (1, 0)
    # d_bottom: (0, -1)
    # d_top: (0, 1)
    
    # We need d/dx (1/d) = -1/d^2 * d/dx(d)
    
    # Create arrays for d^2
    d_left_sq = np.maximum(d_left, 1e-6)**2
    d_right_sq = np.maximum(d_right, 1e-6)**2
    d_bottom_sq = np.maximum(d_bottom, 1e-6)**2
    d_top_sq = np.maximum(d_top, 1e-6)**2
    
    # Gradients
    g_left_x = -1.0 / d_left_sq * -1.0 # -1/d^2 * (-1) = 1/d^2
    g_right_x = -1.0 / d_right_sq * 1.0 # -1/d^2 * (1) = -1/d^2
    g_bottom_y = -1.0 / d_bottom_sq * -1.0 # -1/d^2 * (-1) = 1/d^2
    g_top_y = -1.0 / d_top_sq * 1.0 # -1/d^2 * (1) = -1/d^2
    
    # Select the active wall (min distance)
    # If multiple are min, we can pick any or average. Picking one is standard.
    # Let's create a mask for the min distance.
    all_wall_dists = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_wall_idx = np.argmin(all_wall_dists, axis=1)
    
    wall_grad_x = np.zeros(N)
    wall_grad_y = np.zeros(N)
    
    wall_grad_x[min_wall_idx == 0] = g_left_x[min_wall_idx == 0]
    wall_grad_x[min_wall_idx == 1] = g_right_x[min_wall_idx == 1]
    
    wall_grad_y[min_wall_idx == 2] = g_bottom_y[min_wall_idx == 2]
    wall_grad_y[min_wall_idx == 3] = g_top_y[min_wall_idx == 3]
    
    wall_grad = np.column_stack((wall_grad_x, wall_grad_y))
    
    total_grad = pair_grad + wall_grad
    
    return total_energy, total_grad.flatten()


def _optimize_with_lbfgsb(centres_init, deadline):
    """Optimize using L-BFGS-B on the soft energy function."""
    start_time = time.time()
    
    # Flatten for L-BFGS-B
    x0 = centres_init.flatten()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We want to minimize energy.
    # maxiter is a safety net, but we check time.
    
    # Callback to check time
    def callback(xk):
        if time.time() > deadline - 2:
            # Return True to stop
            return True
        return False
    
    try:
        res = minimize(
            _energy_and_gradient,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or len(res.x) == 2 * N:
            new_centres = res.x.reshape(-1, 2)
            # Ensure finite and valid
            if np.all(np.isfinite(new_centres)):
                return new_centres
    except Exception:
        pass
    
    return centres_init


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # Leave 5s safety margin
    
    best_centres = None
    best_r = -1
    
    # 1. Hexagonal packing as a good starting point
    hex_centres = _hexagonal_packing()
    
    # 2. Fibonacci lattice
    fib_centres = _fibonacci_lattice()
    
    # 3. Random grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    random_grid = grid[:N] + np.random.uniform(-0.01, 0.01, size=(N, 2))
    random_grid = np.clip(random_grid, -0.49, 0.49)
    
    initializations = [
        ("hex", hex_centres),
        ("fib", fib_centres),
        ("rand_grid", random_grid)
    ]
    
    # Add some random perturbations of hex
    for seed in [42, 123]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        perturbed = hex_centres.copy()
        perturbation = np.random.uniform(-0.02, 0.02, size=perturbed.shape)
        perturbed += perturbation
        perturbed = np.clip(perturbed, -0.49, 0.49)
        initializations.append((f"hex_perturb_{seed}", perturbed))
    
    for name, centres in initializations:
        if time.time() > deadline - 10:
            break
        
        # Optimize using L-BFGS-B
        optimized = _optimize_with_lbfgsb(centres, deadline)
        
        # Evaluate radius
        current_r = _compute_radius(optimized)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = optimized.copy()
            
            # If we have a very good result, we can stop early or do a final polish
            # The best known is ~0.0366
            if best_r > 0.0365:
                # Try one more optimization round with the best result as init
                if time.time() < deadline - 5:
                    polished = _optimize_with_lbfgsb(best_centres, deadline)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_r:
                        best_r = polished_r
                        best_centres = polished.copy()
                break
    
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback
    return hex_centres[:N]
