"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n):
    """Initialize circles in a hexagonal lattice pattern inside the unit square."""
    # Hexagonal packing: rows offset by half spacing
    # For n=26, use roughly 6 columns x 5 rows with hex offset
    # Let's try a hexagonal arrangement
    # Row spacing = s * sqrt(3)/2, col spacing = s
    # We want to fit in [0,1]^2
    
    # Try different configurations
    best_init = None
    best_count = 0
    
    for cols in range(4, 9):
        for rows in range(4, 9):
            if cols * rows < n:
                continue
            # Hexagonal: odd rows shifted
            s = 1.0 / (cols - 0.5)  # spacing
            row_h = s * np.sqrt(3) / 2
            # Check if it fits vertically
            if (rows - 1) * row_h + 2 * s > 1.0:
                continue
            # Generate positions
            positions = []
            for r in range(rows):
                for c in range(cols):
                    x = s + c * s + (r % 2) * s / 2
                    y = s + r * row_h
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append((x, y))
            if len(positions) >= n:
                if len(positions) > best_count:
                    best_count = len(positions)
                    best_init = positions[:n]
    
    if best_init is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n)))
        best_init = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:n]
    
    return np.array(best_init)


def _compute_radii(centers):
    """Given centers, compute max possible radii such that circles don't overlap and stay in [0,1]^2."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        d_bound = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        radii[i] = d_bound
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                radii[i] = min(radii[i], d / 2.0)
    return radii


def _energy(centers, radii, centers_opt=None, radii_opt=None):
    """Compute a penalty-based energy for optimization."""
    if centers_opt is not None and radii_opt is not None:
        # We're optimizing both centers and radii
        # Energy = -sum(radii) + penalty for violations
        c = centers_opt
        r = radii_opt
        n = len(c)
        # Penalty for overlap
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < r[i] + r[j]:
                    pen += (r[i] + r[j] - d) ** 2
        # Penalty for going outside
        for i in range(n):
            if c[i, 0] < r[i]:
                pen += (r[i] - c[i, 0]) ** 2
            if c[i, 0] > 1 - r[i]:
                pen += (c[i, 0] - (1 - r[i])) ** 2
            if c[i, 1] < r[i]:
                pen += (r[i] - c[i, 1]) ** 2
            if c[i, 1] > 1 - r[i]:
                pen += (c[i, 1] - (1 - r[i])) ** 2
        return -np.sum(r) + 100.0 * pen
    else:
        # Just compute sum of radii
        return -np.sum(radii)


def _gradient_step(centers, radii, lr, penalty_weight=100.0):
    """Compute gradient of the energy with respect to centers and radii, take a step."""
    n = len(centers)
    dc = np.zeros_like(centers)
    dr = np.zeros_like(radii)
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 1e-10:
                continue
            diff = centers[i] - centers[j]
            unit = diff / d
            # If overlapping, push apart
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                dc[i] += penalty_weight * overlap * unit
                dc[j] -= penalty_weight * overlap * unit
                dr[i] += penalty_weight * overlap
                dr[j] += penalty_weight * overlap
    
    # Boundary constraints
    for i in range(n):
        # x boundaries
        if centers[i, 0] < radii[i]:
            dc[i, 0] -= penalty_weight * (radii[i] - centers[i, 0])
            dr[i] += penalty_weight * (radii[i] - centers[i, 0])
        elif centers[i, 0] > 1 - radii[i]:
            dc[i, 0] += penalty_weight * (centers[i, 0] - (1 - radii[i]))
            dr[i] += penalty_weight * (centers[i, 0] - (1 - radii[i]))
        # y boundaries
        if centers[i, 1] < radii[i]:
            dc[i, 1] -= penalty_weight * (radii[i] - centers[i, 1])
            dr[i] += penalty_weight * (radii[i] - centers[i, 1])
        elif centers[i, 1] > 1 - radii[i]:
            dc[i, 1] += penalty_weight * (centers[i, 1] - (1 - radii[i]))
            dr[i] += penalty_weight * (centers[i, 1] - (1 - radii[i]))
    
    # Gradient of -sum(radii) is -1 for each radius
    dr -= 1.0
    
    # Apply gradient descent step
    centers_new = centers - lr * dc
    radii_new = radii - lr * dr
    
    # Clamp radii to be positive
    radii_new = np.maximum(radii_new, 1e-8)
    
    # Clamp centers to stay in [0,1]
    centers_new = np.clip(centers_new, 0.0, 1.0)
    
    return centers_new, radii_new


def construct():
    """Pack 26 circles in the unit square, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with hexagonal pattern
    centers = _hex_init(N)
    radii = _compute_radii(centers)
    
    best_score = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Simulated annealing / gradient-based optimization
    n_iter = 50000
    lr = 0.01
    cooling = 0.99995
    
    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Adaptive learning rate
        current_lr = lr * (cooling ** it)
        
        # Take gradient step
        centers_new, radii_new = _gradient_step(centers, radii, current_lr)
        
        # Check if improvement
        new_score = np.sum(radii_new)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers_new.copy()
            best_radii = radii_new.copy()
        
        centers = centers_new
        radii = radii_new
        
        # Occasionally add small random perturbation to escape local minima
        if it % 1000 == 0 and it > 0:
            centers += np.random.normal(0, 0.01, centers.shape)
            centers = np.clip(centers, 0.0, 1.0)
    
    # Final cleanup: ensure no overlaps by shrinking radii slightly
    final_radii = _compute_radii(best_centers)
    final_score = np.sum(final_radii)
    
    # Use the better of the optimized radii or the computed max radii
    if final_score > best_score:
        best_radii = final_radii
        best_score = final_score
    
    # Build output array
    result = np.column_stack([best_centers, best_radii])
    return result
# EVOLVE-BLOCK-END
