"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for a set of centers."""
    # Distance between all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to boundaries
    # Square is [-0.5, 0.5]^2
    boundary_dist = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1]
    ], axis=1), axis=1)
    min_boundary_dist = np.min(boundary_dist)
    
    return min(min_pair_dist, min_boundary_dist) / 2.0


def _energy(centers, target_r):
    """
    Soft energy function: penalize pairs closer than 2*target_r and points outside the square.
    Returns a scalar energy value.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pairwise penalty: smooth hinge for dist < 2*target_r
    # Using (2*r - d)^2 when d < 2*r, else 0
    violation = np.maximum(2.0 * target_r - dist, 0.0)
    pair_energy = np.sum(violation ** 2)
    
    # Boundary penalty: points should be in [-0.5 + target_r, 0.5 - target_r]
    # Penalty for x: max(0, (0.5 - target_r) - x) + max(0, x - (-0.5 + target_r))
    x_pen = np.maximum(0.0, (0.5 - target_r) - centers[:, 0]) + \
            np.maximum(0.0, centers[:, 0] - (-0.5 + target_r))
    y_pen = np.maximum(0.0, (0.5 - target_r) - centers[:, 1]) + \
            np.maximum(0.0, centers[:, 1] - (-0.5 + target_r))
    boundary_energy = np.sum(x_pen ** 2) + np.sum(y_pen ** 2)
    
    return pair_energy + boundary_energy


def _energy_grad(centers, target_r):
    """
    Gradient of the energy function with respect to centers.
    Returns array of shape (N, 2).
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-10, dist, 1e-10)
    
    # Pairwise gradient
    # d/dx_i sum_j max(2r - |x_i - x_j|, 0)^2
    # = sum_j 2 * max(2r - d_ij, 0) * (-1) * (x_i - x_j) / d_ij
    # = -2 * sum_j max(2r - d_ij, 0) * (x_i - x_j) / d_ij
    
    viol = np.maximum(2.0 * target_r - dist, 0.0)
    # Gradient contribution from each pair to point i
    # grad_i = -2 * sum_{j != i} viol_ij * (x_i - x_j) / d_ij
    factor = 2.0 * viol / dist_safe
    # Set diagonal to 0
    np.fill_diagonal(factor, 0.0)
    pair_grad = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary gradient
    x_pen_grad = np.zeros_like(centers[:, 0])
    y_pen_grad = np.zeros_like(centers[:, 1])
    
    # x: penalty = max(0, (0.5 - r) - x)^2 + max(0, x - (-0.5 + r))^2
    # grad = 2*max(0, (0.5-r)-x)*(-1) + 2*max(0, x-(-0.5+r))*(1)
    x_pen_grad += -2.0 * np.maximum(0.0, (0.5 - target_r) - centers[:, 0])
    x_pen_grad += 2.0 * np.maximum(0.0, centers[:, 0] - (-0.5 + target_r))
    
    y_pen_grad += -2.0 * np.maximum(0.0, (0.5 - target_r) - centers[:, 1])
    y_pen_grad += 2.0 * np.maximum(0.0, centers[:, 1] - (-0.5 + target_r))
    
    grad = np.column_stack([pair_grad[:, 0] + x_pen_grad, 
                            pair_grad[:, 1] + y_pen_grad])
    return grad


def _initial_grid():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _initial_hexagonal():
    """Create a hexagonal lattice initialization."""
    # Hexagonal packing in a unit square
    # For hexagonal packing, rows are offset
    # Approximate number of rows and columns
    # In hexagonal packing, horizontal spacing is 2r, vertical is sqrt(3)*r
    # We want to fit ~200 points
    # Let's try a grid with hexagonal offset
    
    # Start with a grid and add small offsets
    m = int(np.ceil(np.sqrt(N)))
    centers = []
    for row in range(m):
        for col in range(m):
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            # Offset every other row
            if row % 2 == 1:
                x += 0.5 / m
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    return centers


def _initial_random():
    """Create a random initialization inside the square."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _local_search(centers, target_r, max_iter=2000, lr=0.001, deadline=None):
    """
    Perform gradient descent to minimize energy for a given target radius.
    Returns the final centers.
    """
    centers = centers.copy()
    t0 = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e = _energy(centers, target_r)
        if e < 1e-10:
            break
        
        grad = _energy_grad(centers, target_r)
        centers -= lr * grad
        
        # Project back into the square (with some margin)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Adaptive learning rate: decrease if energy increases
        e_new = _energy(centers, target_r)
        if e_new > e * 1.01:
            lr *= 0.5
            centers += lr * grad  # Undo the step
            e_new = _energy(centers, target_r)
        
        # Occasionally increase lr if stuck
        if it > 100 and e_new < 1e-8:
            lr = min(lr * 1.1, 0.01)
    
    return centers


def _try_target_radius(centers_init, target_r, max_iter=2000, deadline=None):
    """
    Try to achieve a packing with minimum radius >= target_r.
    Returns (success, centers).
    """
    centers = _local_search(centers_init, target_r, max_iter=max_iter, deadline=deadline)
    
    # Check if the achieved radius is at least target_r
    achieved_r = _compute_radius(centers)
    if achieved_r >= target_r - 1e-9:
        return True, centers
    return False, centers


def construct():
    """
    Construct the packing by trying to maximize the common radius.
    Uses binary search on the target radius with local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial radii bounds
    # Grid gives about 1/(2*sqrt(200)) ~ 0.035
    # Best known is ~0.0366
    r_low = 0.035
    r_high = 0.038
    
    # Try multiple initializations
    initializations = [
        _initial_grid(),
        _initial_hexagonal(),
        _initial_random(),
        _initial_random(),
    ]
    
    best_centers = None
    best_radius = 0.0
    
    # First, evaluate the initializations
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Binary search for the maximum achievable radius
    # We'll try to improve upon the best initial radius
    if time.time() > deadline - 5.0:
        return best_centers
    
    # Try to find a better radius via binary search
    # Start with a modest target and increase
    current_target = best_radius + 0.0005
    
    # For each target, try all initializations and keep the best
    while time.time() < deadline - 10.0:
        # Try to achieve current_target
        found = False
        for init in initializations:
            if time.time() > deadline - 10.0:
                break
            
            success, centers = _try_target_radius(init, current_target, 
                                                  max_iter=1500, 
                                                  deadline=deadline - 5.0)
            if success:
                found = True
                r = _compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
                break
        
        if found:
            current_target += 0.0002
        else:
            # Try a slightly lower target
            current_target -= 0.0001
            if current_target <= best_radius:
                break
    
    # Final refinement: try to push the radius a bit more
    if best_centers is not None and time.time() < deadline - 5.0:
        # Try local search with the current best radius as target
        refined = _local_search(best_centers, best_radius, max_iter=500, 
                               deadline=deadline - 1.0)
        r_refined = _compute_radius(refined)
        if r_refined > best_radius:
            best_radius = r_refined
            best_centers = refined
    
    # Ensure all centers are strictly inside the square
    if best_centers is not None:
        best_centers[:, 0] = np.clip(best_centers[:, 0], -0.499, 0.499)
        best_centers[:, 1] = np.clip(best_centers[:, 1], -0.499, 0.499)
    
    return best_centers
# EVOLVE-BLOCK-END
