"""Circle packing in unit square n=32 - structural optimization via force-directed relaxation."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay

N = 32
DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _initial_hex_layout(n):
    """Generate a hexagonal close-packing initial layout for n circles."""
    # Hexagonal packing: rows offset
    # For 32 circles, we can do roughly 6x6 grid with hex offset
    # Let's generate a hex grid and pick the first n positions
    positions = []
    r = 0.5 / 6  # approximate radius for 6x6
    dx = 2 * r
    dy = np.sqrt(3) * r
    for row in range(10):
        for col in range(10):
            x = dx * (col + 0.5 * (row % 2)) + 0.05
            y = dy * row + 0.05
            positions.append((x, y))
    # Shuffle and take first n, then sort for better initial structure
    positions = positions[:n]
    return np.array(positions)


def _initial_grid_layout(n):
    """Generate a grid layout with slight randomization."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side + 0.01 * np.random.randn()
                y = (j + 0.5) / side + 0.01 * np.random.randn()
                positions.append((x, y))
    return np.array(positions[:n])


def _compute_radii(positions):
    """Compute maximum radii for given positions such that circles are disjoint and inside [0,1]^2."""
    n = len(positions)
    # Distance to boundary for each circle
    dist_to_boundary = np.minimum(
        np.minimum(positions[:, 0], 1.0 - positions[:, 0]),
        np.minimum(positions[:, 1], 1.0 - positions[:, 1])
    )
    # Distance to other circles
    radii = np.zeros(n)
    for i in range(n):
        min_dist = dist_to_boundary[i]
        for j in range(n):
            if i != j:
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d / 2.0)
        radii[i] = max(0, min_dist - 1e-9)
    return radii


def _objective(x):
    """Objective: minimize negative sum of radii."""
    n = N
    positions = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Penalty for circles going outside [0,1]^2
    penalty = 0.0
    for i in range(n):
        x_i, y_i = positions[i]
        r_i = radii[i]
        if x_i - r_i < 0 or x_i + r_i > 1 or y_i - r_i < 0 or y_i + r_i > 1:
            penalty += 100.0 * max(0, r_i - x_i) + max(0, r_i - (1-x_i)) + max(0, r_i - y_i) + max(0, r_i - (1-y_i))
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j]
            if d < min_dist:
                penalty += 100.0 * (min_dist - d)
    
    return -np.sum(radii) + penalty


def _solve_with_slsqp(positions_init, radii_init, time_budget):
    """Use SLSQP to optimize the packing."""
    n = N
    x0 = np.concatenate([positions_init.flatten(), radii_init])
    
    # Bounds: positions in [0,1], radii >= 0
    bounds = []
    for i in range(2*n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
    
    # Constraints: circle i must be inside [0,1]^2
    # x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    # r_i + r_j + 1e-9 <= ||p_i - p_j|| for all i < j
    # SLSQP handles non-linear constraints, but let's use penalty method instead for robustness
    
    start_time = time.time()
    best_x = x0.copy()
    best_val = _objective(x0)
    
    result = minimize(
        _objective, x0, method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
    )
    
    if result.success or result.fun < best_val:
        best_x = result.x
        best_val = result.fun
    
    elapsed = time.time() - start_time
    return best_x, elapsed


def _force_directed_relaxation(positions, max_iter=5000, time_budget=10.0):
    """Iterative force-directed relaxation to improve packing."""
    n = N
    positions = positions.copy()
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        radii = _compute_radii(positions)
        if np.sum(radii) < 1e-12:
            break
        
        # Compute forces: each circle wants to push away from others
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(i+1, n):
                d_vec = positions[i] - positions[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    d_vec = np.random.randn(2) * 1e-5
                    d = np.linalg.norm(d_vec)
                min_dist = radii[i] + radii[j]
                if d < min_dist + 0.01:
                    # Repulsive force
                    strength = (min_dist + 0.01 - d) / d
                    forces[i] += strength * d_vec
                    forces[j] -= strength * d_vec
            
            # Attractive force toward center to keep circles together
            center = np.array([0.5, 0.5])
            to_center = center - positions[i]
            forces[i] += 0.01 * to_center
        
        # Update positions with adaptive step size
        step = 0.001
        new_positions = positions + step * forces
        # Clamp to [0,1]
        new_positions = np.clip(new_positions, 0.0, 1.0)
        
        # Check if improvement
        new_radii = _compute_radii(new_positions)
        if np.sum(new_radii) > np.sum(radii):
            positions = new_positions
        else:
            step *= 0.5
    
    return positions


def _multi_start_optimize(time_budget):
    """Try multiple initial layouts and optimize each."""
    best_positions = None
    best_radii = None
    best_sum = -1
    
    start_time = time.time()
    
    # Try different initial layouts
    initializations = [
        _initial_grid_layout,
        _initial_hex_layout,
        lambda n: np.random.rand(n, 2),  # random
        lambda n: np.random.rand(n, 2) * 0.8 + 0.1,  # random with margin
    ]
    
    for init_func in initializations:
        if time.time() - start_time > time_budget:
            break
        
        positions = init_func(N)
        radii = _compute_radii(positions)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_positions = positions.copy()
            best_radii = radii.copy()
        
        # Force-directed relaxation
        remaining = time_budget - (time.time() - start_time)
        if remaining > 2:
            positions = _force_directed_relaxation(positions, time_budget=min(5, remaining - 1))
            radii = _compute_radii(positions)
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # SLSQP optimization
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5:
            x, elapsed = _solve_with_slsqp(positions, radii, min(10, remaining - 2))
            pos = x[:2*N].reshape(N, 2)
            rad = x[2*N:]
            # Verify validity
            valid = True
            for i in range(N):
                if pos[i][0] - rad[i] < -1e-6 or pos[i][0] + rad[i] > 1 + 1e-6 or \
                   pos[i][1] - rad[i] < -1e-6 or pos[i][1] + rad[i] > 1 + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.linalg.norm(pos[i] - pos[j])
                        if d < rad[i] + rad[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                current_sum = np.sum(rad)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_positions = pos.copy()
                    best_radii = rad.copy()
    
    return best_positions, best_radii


def construct():
    """Construct 32 disjoint circles in [0,1]^2 maximizing sum of radii."""
    np.random.seed(42)
    
    start_time = time.time()
    
    # Multi-start optimization
    positions, radii = _multi_start_optimize(time_budget=DEADLINE - 10)
    
    # Final verification and cleanup
    if positions is None or radii is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Ensure circles are strictly inside and disjoint
    result = np.zeros((N, 3))
    for i in range(N):
        x, y = positions[i]
        r = radii[i]
        # Clamp to ensure inside [0,1]^2
        r = min(r, x, 1-x, y, 1-y)
        r = max(0, r - 1e-7)
        result[i] = [x, y, r]
    
    # Final check for overlaps, shrink if needed
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i,:2] - result[j,:2])
            min_dist = result[i,2] + result[j,2]
            if d < min_dist:
                # Shrink both
                scale = (d - 1e-8) / min_dist
                if scale < 1:
                    result[i,2] *= scale
                    result[j,2] *= scale
    
    return result

# EVOLVE-BLOCK-END
