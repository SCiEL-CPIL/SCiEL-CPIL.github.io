"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 21

# EVOLVE-BLOCK-START

def _validate_and_compute(radii, x, y, W, H):
    """Check validity and compute sum of radii. Returns (is_valid, value)."""
    n = len(radii)
    # Check bounds
    for i in range(n):
        if x[i] < radii[i] - 1e-9 or x[i] > W - radii[i] + 1e-9:
            return False, 0.0
        if y[i] < radii[i] - 1e-9 or y[i] > H - radii[i] + 1e-9:
            return False, 0.0
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return False, 0.0
    # Check perimeter
    perim = 2*(W + H)
    if perim > 4.0 + 1e-9:
        return False, 0.0
    value = np.sum(radii)
    return True, value


def _seed_grid():
    """Create a 5x5 grid of circles in a 1x1 square (perimeter 4)."""
    side = 5
    W = 1.0
    H = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:N])


def _optimize(initial, deadline):
    """Local optimization using gradient-free search on positions and radii."""
    n = len(initial)
    best_x = initial[:, 0].copy()
    best_y = initial[:, 1].copy()
    best_r = initial[:, 2].copy()
    W = 1.0
    H = 1.0
    _, best_val = _validate_and_compute(best_r, best_x, best_y, W, H)
    
    # Simple hill climbing with random perturbations
    step_size = 0.05
    for iter in range(2000):
        if time.time() > deadline:
            break
        # Try random perturbation
        x_new = best_x.copy()
        y_new = best_y.copy()
        r_new = best_r.copy()
        
        idx = np.random.randint(n)
        x_new[idx] += np.random.uniform(-step_size, step_size)
        y_new[idx] += np.random.uniform(-step_size, step_size)
        r_new[idx] = max(0.01, r_new[idx] + np.random.uniform(-0.01, 0.01))
        
        # Clamp to bounds
        x_new[idx] = np.clip(x_new[idx], r_new[idx], W - r_new[idx])
        y_new[idx] = np.clip(y_new[idx], r_new[idx], H - r_new[idx])
        
        is_valid, val = _validate_and_compute(r_new, x_new, y_new, W, H)
        if is_valid and val > best_val:
            best_x = x_new
            best_y = y_new
            best_r = r_new
            best_val = val
            step_size *= 0.98
        else:
            step_size = min(0.1, step_size * 1.02)
    
    return np.column_stack([best_x, best_y, best_r]), best_val


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with grid
    initial = _seed_grid()
    result, val = _optimize(initial, deadline)
    
    # Try a few more restarts if time permits
    for restart in range(5):
        if time.time() > deadline:
            break
        # Perturb initial grid
        perturbed = initial.copy()
        for i in range(len(perturbed)):
            perturbed[i, 0] += np.random.uniform(-0.05, 0.05)
            perturbed[i, 1] += np.random.uniform(-0.05, 0.05)
            perturbed[i, 2] = max(0.01, perturbed[i, 2] + np.random.uniform(-0.02, 0.02))
        
        r_new, v_new = _optimize(perturbed, deadline)
        if v_new > val:
            result = r_new
            val = v_new
    
    return result
# EVOLVE-BLOCK-END
