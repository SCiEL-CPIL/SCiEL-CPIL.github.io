"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START

def _hex_init(n):
    """
    Create a hexagonal packing of n circles in the unit square.
    We arrange circles in rows with alternating offsets (hexagonal close packing).
    We choose a number of rows and columns that approximately fits n circles.
    For n=32, a 5x7 or 6x6 arrangement works.
    We use a 6-row x 6-col grid but with hexagonal offset, then trim/pad to n.
    Actually, let's use a more systematic approach:
    Place circles in a hexagonal lattice, scale to fit in unit square, take first n.
    """
    # Hexagonal lattice parameters
    # For a hexagonal lattice, if we have 'rows' rows and 'cols' columns,
    # the spacing in x is dx, in y is dy = dx * sqrt(3)/2
    # We want to fit as many as possible, then take n.
    
    # Let's try a 6x6 grid with hexagonal offset
    rows = 6
    cols = 6
    # Total circles in this grid
    total = rows * cols
    # We'll generate all positions, then scale to fit in [0,1]^2
    # and take the first n (or all if total < n)
    
    # Hexagonal lattice positions (before scaling)
    # Row i, col j:
    # x = j * 1.0 + (i % 2) * 0.5
    # y = i * sqrt(3)/2
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # Scale to fit in [0, 1]^2
    x_min, y_min = positions[:, 0].min(), positions[:, 1].min()
    x_max, y_max = positions[:, 0].max(), positions[:, 1].max()
    
    # We want the bounding box to be [0, 1]^2
    # But we need to account for circle radii, so we'll scale slightly smaller
    # For now, scale to [0.05, 0.95] to leave room for radii
    margin = 0.05
    scale_x = (1 - 2 * margin) / (x_max - x_min + 1e-9)
    scale_y = (1 - 2 * margin) / (y_max - y_min + 1e-9)
    scale = min(scale_x, scale_y)
    
    positions[:, 0] = (positions[:, 0] - x_min) * scale + margin
    positions[:, 1] = (positions[:, 1] - y_min) * scale + margin
    
    # Initial radius estimate
    # The minimum distance between adjacent circles
    # For hexagonal lattice, nearest neighbor distance is 1.0 in the original lattice
    # After scaling, it's scale * 1.0
    # Radius should be half the minimum distance
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, d)
    r_init = min_dist / 2 * 0.95  # slightly smaller to ensure no overlap
    
    # Create circles
    circles = []
    for i in range(min(n, len(positions))):
        circles.append([positions[i, 0], positions[i, 1], r_init])
    
    # If we have fewer than n circles, add more randomly
    if len(circles) < n:
        for i in range(len(circles), n):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            circles.append([x, y, r_init * 0.5])
    
    return np.array(circles[:n])


def _compute_objective(circles):
    """
    Compute the objective: sum of radii, but penalize overlaps and boundary violations.
    Returns (objective, is_valid).
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check boundary violations
    boundary_violation = 0.0
    for i in range(len(circles)):
        if x[i] < r[i]:
            boundary_violation += (r[i] - x[i])
        if x[i] > 1 - r[i]:
            boundary_violation += (x[i] - (1 - r[i]))
        if y[i] < r[i]:
            boundary_violation += (r[i] - y[i])
        if y[i] > 1 - r[i]:
            boundary_violation += (y[i] - (1 - r[i]))
    
    # Check overlaps
    overlap_penalty = 0.0
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist)
    
    penalty = boundary_violation + overlap_penalty
    objective = np.sum(r) - 100 * penalty
    
    is_valid = boundary_violation < 1e-8 and overlap_penalty < 1e-8
    return objective, is_valid, np.sum(r)


def _push_apart(circles, iterations=100, strength=0.1):
    """
    Push overlapping circles apart and pull boundary-violating circles inward.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(circles)
    
    for it in range(iterations):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i]))
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-9:
                    # Push apart
                    force = (min_dist - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                elif dist <= 1e-9:
                    # Random push
                    fx[i] += np.random.uniform(-0.1, 0.1)
                    fy[i] += np.random.uniform(-0.1, 0.1)
                    fx[j] -= np.random.uniform(-0.1, 0.1)
                    fy[j] -= np.random.uniform(-0.1, 0.1)
        
        # Apply forces
        x += strength * fx
        y += strength * fy
        
        # Clamp to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Update circles
        circles[:, 0] = x
        circles[:, 1] = y
    
    return circles


def _grow_radii(circles, max_iterations=200, growth_rate=0.001):
    """
    Try to grow radii while maintaining feasibility.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(circles)
    
    for it in range(max_iterations):
        # Try to grow each radius slightly
        improved = False
        for i in range(n):
            r_new = r[i] + growth_rate
            # Check if this maintains feasibility
            # Check boundary
            if r_new > x[i] or r_new > 1 - x[i] or r_new > y[i] or r_new > 1 - y[i]:
                continue
            # Check overlaps
            valid = True
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r_new + r[j]:
                    valid = False
                    break
            if valid:
                r[i] = r_new
                improved = True
        
        if not improved:
            break
    
    circles[:, 2] = r
    return circles


def _local_search(circles, n_restarts=5, max_iterations=500):
    """
    Local search: try to improve the packing by adjusting positions.
    """
    best_circles = circles.copy()
    best_obj, best_valid, best_sum_r = _compute_objective(best_circles)
    
    for restart in range(n_restarts):
        current = best_circles.copy()
        # Add small random perturbation
        if restart > 0:
            current[:, :2] += np.random.normal(0, 0.01, size=(len(current), 2))
            current[:, :2] = np.clip(current[:, :2], 0, 1)
        
        for it in range(max_iterations):
            # Pick a random circle and try to move it
            i = np.random.randint(len(current))
            dx = np.random.normal(0, 0.01)
            dy = np.random.normal(0, 0.01)
            
            new_x = current[i, 0] + dx
            new_y = current[i, 1] + dy
            
            # Check bounds
            if new_x < current[i, 2] or new_x > 1 - current[i, 2]:
                continue
            if new_y < current[i, 2] or new_y > 1 - current[i, 2]:
                continue
            
            # Check overlaps
            valid = True
            for j in range(len(current)):
                if i == j:
                    continue
                ddx = new_x - current[j, 0]
                ddy = new_y - current[j, 1]
                dist = np.sqrt(ddx * ddx + ddy * ddy)
                if dist < current[i, 2] + current[j, 2]:
                    valid = False
                    break
            
            if valid:
                current[i, 0] = new_x
                current[i, 1] = new_y
        
        # Push apart and grow
        current = _push_apart(current, iterations=50, strength=0.05)
        current = _grow_radii(current, max_iterations=100, growth_rate=0.0005)
        
        obj, valid, sum_r = _compute_objective(current)
        if valid and sum_r > best_sum_r:
            best_circles = current.copy()
            best_sum_r = sum_r
    
    return best_circles


def construct():
    """
    Pack 32 circles in the unit square to maximize the sum of radii.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to leave safety margin
    
    # Initialize with hexagonal packing
    np.random.seed(42)
    circles = _hex_init(N)
    
    # Push apart to resolve any initial overlaps
    circles = _push_apart(circles, iterations=200, strength=0.1)
    
    # Grow radii
    circles = _grow_radii(circles, max_iterations=500, growth_rate=0.002)
    
    # Local search
    best_circles = _local_search(circles, n_restarts=3, max_iterations=300)
    
    # Check time
    if time.time() > deadline - 20:
        # Final polish: push apart and grow
        best_circles = _push_apart(best_circles, iterations=100, strength=0.05)
        best_circles = _grow_radii(best_circles, max_iterations=200, growth_rate=0.001)
        return best_circles
    
    # Try more local search if time permits
    while time.time() < deadline - 30:
        # Perturb and improve
        perturbed = best_circles.copy()
        perturbed[:, :2] += np.random.normal(0, 0.02, size=(N, 2))
        perturbed[:, :2] = np.clip(perturbed[:, :2], 0, 1)
        
        perturbed = _push_apart(perturbed, iterations=100, strength=0.05)
        perturbed = _grow_radii(perturbed, max_iterations=200, growth_rate=0.001)
        
        obj, valid, sum_r = _compute_objective(perturbed)
        _, _, best_sum_r = _compute_objective(best_circles)
        
        if valid and sum_r > best_sum_r:
            best_circles = perturbed
    
    # Final polish
    best_circles = _push_apart(best_circles, iterations=200, strength=0.05)
    best_circles = _grow_radii(best_circles, max_iterations=500, growth_rate=0.001)
    
    # Ensure validity
    _, valid, _ = _compute_objective(best_circles)
    if not valid:
        # Try to fix
        best_circles = _push_apart(best_circles, iterations=500, strength=0.1)
        # Shrink radii slightly if still invalid
        _, valid, _ = _compute_objective(best_circles)
        if not valid:
            best_circles[:, 2] *= 0.95
    
    return best_circles

# EVOLVE-BLOCK-END
