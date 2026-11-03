"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Deadline guard
_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)


def _radius(positions):
    """Compute the radius for a given set of positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to walls
    wall_dists = []
    for i in range(n):
        x, y = positions[i]
        wall_dists.append(min(x + 0.5, 0.5 - x, y + 0.5, 0.5 - y))
    min_wall = min(wall_dists)
    # Pairwise distances
    min_pair = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_pair:
                min_pair = d
    min_pair_half = min_pair / 2.0
    return min(min_wall, min_pair_half)


def _radius_fast(positions):
    """Faster radius computation using vectorized operations."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.minimum.reduce([
        positions[:, 0] + 0.5,
        0.5 - positions[:, 0],
        positions[:, 1] + 0.5,
        0.5 - positions[:, 1]
    ])
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_half = np.min(dists) / 2.0
    return min(min_wall, min_pair_half)


def _soft_min_radius(positions, alpha=10.0):
    """Soft-min approximation of the radius using log-sum-exp."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to walls
    wall_dists = np.minimum.reduce([
        positions[:, 0] + 0.5,
        0.5 - positions[:, 0],
        positions[:, 1] + 0.5,
        0.5 - positions[:, 1]
    ])
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[~np.eye(n, dtype=bool)]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    # Soft min using log-sum-exp
    min_d = np.min(all_dists)
    if min_d < 1e-10:
        return 0.0
    log_sum = np.log(np.sum(np.exp(-alpha * (all_dists - min_d))))
    return min_d - np.log(log_sum) / alpha


def _soft_min_radius_neg(positions, alpha=10.0):
    """Negative soft-min for minimization."""
    return -_soft_min_radius(positions, alpha)


def _hard_disk_expansion(positions, max_iter=500, step_scale=1.0):
    """Expand circles using hard-disk repulsion."""
    n = len(positions)
    positions = positions.copy()
    best_r = _radius(positions)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        # Compute forces
        forces = np.zeros_like(positions)
        r = _radius(positions)
        if r <= 1e-10:
            break
        # Wall forces
        for i in range(n):
            x, y = positions[i]
            # Left wall
            if x + 0.5 < r:
                forces[i, 0] += (r - (x + 0.5))
            # Right wall
            if 0.5 - x < r:
                forces[i, 0] -= (r - (0.5 - x))
            # Bottom wall
            if y + 0.5 < r:
                forces[i, 1] += (r - (y + 0.5))
            # Top wall
            if 0.5 - y < r:
                forces[i, 1] -= (r - (0.5 - y))
        
        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r and dist > 1e-10:
                    overlap = 2 * r - dist
                    forces[i] += overlap * d / dist
                    forces[j] -= overlap * d / dist
        
        # Update positions
        step = 0.01 * step_scale
        positions = positions + step * forces
        # Clip to stay within bounds
        positions[:, 0] = np.clip(positions[:, 0], -0.5 + r, 0.5 - r)
        positions[:, 1] = np.clip(positions[:, 1], -0.5 + r, 0.5 - r)
        
        new_r = _radius(positions)
        if new_r > best_r:
            best_r = new_r
    
    return positions, best_r


def _slsqp_polish(positions, max_iter=200):
    """Polish using SLSQP with hard constraints."""
    n = len(positions)
    x0 = positions.flatten()
    
    # Objective: maximize radius = minimize -radius
    def objective(x):
        pos = x.reshape(n, 2)
        # Check bounds
        if np.any(pos < -0.5) or np.any(pos > 0.5):
            return 1.0
        return -_radius(pos)
    
    # Constraints: keep within square
    def bounds():
        return [(-0.5, 0.5)] * (2 * n)
    
    try:
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds(), 
                         options={'maxiter': max_iter, 'ftol': 1e-10})
        if result.success or result.fun < objective(x0):
            pos = result.x.reshape(n, 2)
            r = _radius(pos)
            if r > 0:
                return pos, r
    except Exception:
        pass
    
    return positions, _radius(positions)


def construct():
    """Construct 50 circle centers in unit square [-0.5, 0.5]^2."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with a reasonable initial configuration
    # Use a hexagonal packing pattern as initial guess
    n = N
    # Create a grid-like initial position
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    positions = grid[:n].copy()
    
    best_positions = positions.copy()
    best_r = _radius(positions)
    
    # Phase 1: Hard-disk expansion
    if _time_left() > 20:
        positions, r = _hard_disk_expansion(positions, max_iter=300)
        if r > best_r:
            best_r = r
            best_positions = positions.copy()
    
    # Phase 2: SLSQP polish
    if _time_left() > 10:
        positions, r = _slsqp_polish(best_positions, max_iter=100)
        if r > best_r:
            best_r = r
            best_positions = positions.copy()
    
    # Phase 3: Additional polish iterations if time allows
    while _time_left() > 5:
        positions, r = _slsqp_polish(best_positions, max_iter=50)
        if r > best_r:
            best_r = r
            best_positions = positions.copy()
        else:
            break
    
    # Validate
    if not np.all(np.isfinite(best_positions)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    # Ensure positions are within bounds
    best_positions[:, 0] = np.clip(best_positions[:, 0], -0.5, 0.5)
    best_positions[:, 1] = np.clip(best_positions[:, 1], -0.5, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
