"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np


def _pairwise_constraints(x, y, r):
    """Check if all pairwise distances are at least r_i + r_j."""
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-12:
                return False
    return True


def _is_valid(x, y, r, tol=1e-9):
    """Check validity: circles inside [0,1]^2 and disjoint interiors."""
    n = len(x)
    # Check bounds
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-10:
                return False
    return True


def _score(x, y, r):
    """Sum of radii."""
    return float(np.sum(r))


def _make_grid_initial(n):
    """Create a grid initial placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y, r])
    return np.array(positions[:n])


def _make_hex_initial(n):
    """Create a hexagonal packing initial placement."""
    positions = []
    # Hexagonal grid
    row_height = 1.0 / (n + 2)
    col_width = row_height * 2 / np.sqrt(3)
    y = row_height * 0.5
    row = 0
    while len(positions) < n:
        x_offset = 0.0
        if row % 2 == 1:
            x_offset = col_width * 0.5
        x = x_offset + col_width * 0.5
        col = 0
        while len(positions) < n and x <= 1.0:
            r = row_height * 0.5 - 1e-4
            positions.append([x, y, r])
            x += col_width
            col += 1
        y += row_height
        row += 1
    return np.array(positions[:n])


def _make_random_initial(n, seed):
    """Create a random initial placement."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.01 + rng.random() * 0.05
        positions.append([x, y, r])
    return np.array(positions[:n])


def _relax_and_grow(positions, max_iters=500, tol=1e-8):
    """
    Iteratively relax overlaps and grow radii.
    Returns improved positions.
    """
    n = len(positions)
    pos = positions.copy()
    
    for it in range(max_iters):
        # Check for overlaps and resolve
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pos[i][2] + pos[j][2]
                
                if dist < min_dist - 1e-12:
                    # Overlap: push apart
                    if dist < 1e-12:
                        # Random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    # Push each circle by half the overlap
                    pos[i][0] += dx * overlap * 0.5
                    pos[i][1] += dy * overlap * 0.5
                    pos[j][0] -= dx * overlap * 0.5
                    pos[j][1] -= dy * overlap * 0.5
                    moved = True
        
        # Clamp to unit square
        for i in range(n):
            pos[i][0] = np.clip(pos[i][0], pos[i][2], 1 - pos[i][2])
            pos[i][1] = np.clip(pos[i][1], pos[i][2], 1 - pos[i][2])
        
        # Try to grow radii
        max_growth = 0
        for i in range(n):
            # Compute max possible radius for circle i
            max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = (dist - pos[j][2])
                max_r = min(max_r, available)
            max_r = max(max_r, 0)
            if max_r > pos[i][2] + 1e-10:
                growth = (max_r - pos[i][2]) * 0.1
                pos[i][2] += growth
                max_growth = max(max_growth, growth)
        
        # Clamp radii to be valid
        for i in range(n):
            pos[i][2] = min(pos[i][2], pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        
        if max_growth < tol and not moved:
            break
    
    return pos


def _local_search(positions, max_iters=200, tol=1e-8):
    """
    Local search: try small perturbations and accept if score improves.
    """
    n = len(positions)
    best_pos = positions.copy()
    best_score = _score(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2])
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            # Try moving circle i in random directions
            for _ in range(20):
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(0.001, 0.01)
                new_pos = best_pos.copy()
                new_pos[i][0] += step * np.cos(angle)
                new_pos[i][1] += step * np.sin(angle)
                
                # Clamp
                new_pos[i][0] = np.clip(new_pos[i][0], new_pos[i][2], 1 - new_pos[i][2])
                new_pos[i][1] = np.clip(new_pos[i][1], new_pos[i][2], 1 - new_pos[i][2])
                
                if _is_valid(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]):
                    new_score = _score(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2])
                    if new_score > best_score + 1e-10:
                        best_pos = new_pos
                        best_score = new_score
                        improved = True
        
        if not improved:
            break
    
    return best_pos


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_make_grid_initial(N))
    inits.append(_make_hex_initial(N))
    
    # Add random inits
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        inits.append(_make_random_initial(N, seed))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        
        # Relax and grow
        relaxed = _relax_and_grow(init, max_iters=300)
        
        # Local search
        searched = _local_search(relaxed, max_iters=100)
        
        score = _score(searched[:, 0], searched[:, 1], searched[:, 2])
        
        if _is_valid(searched[:, 0], searched[:, 1], searched[:, 2]):
            if score > best_score:
                best_score = score
                best_result = searched
    
    # Final polish: try to grow radii more aggressively
    if best_result is not None and time.time() < deadline - 2:
        polished = _relax_and_grow(best_result, max_iters=200, tol=1e-10)
        polished_score = _score(polished[:, 0], polished[:, 1], polished[:, 2])
        if _is_valid(polished[:, 0], polished[:, 1], polished[:, 2]) and polished_score > best_score:
            best_result = polished
            best_score = polished_score
    
    if best_result is None:
        # Fallback to grid
        best_result = _make_grid_initial(N)
    
    return best_result
# EVOLVE-BLOCK-END
