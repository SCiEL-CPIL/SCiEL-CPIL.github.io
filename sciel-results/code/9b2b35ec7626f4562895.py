"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _radius(c):
    """Compute the common radius for a set of centres."""
    # Distance between all pairs
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = wall_dists.min()
    return min(min_pair_dist / 2.0, min_wall_dist)

def _score(c):
    """Return the score (larger is better)."""
    r = _radius(c)
    # Check validity: all centres must be inside the square
    if not np.all(np.abs(c) <= 0.5):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    return r

def _make_honeycomb():
    """Create a hexagonal/honeycomb packing pattern."""
    # For 50 circles, use a hexagonal grid
    # Hexagonal packing has rows offset
    # Let's try a few configurations and pick the best
    best_c = None
    best_r = 0
    
    # Try different row counts
    for rows in range(5, 11):
        # Distribute 50 circles across rows
        base = N // rows
        extra = N % rows
        row_sizes = [base + (1 if i < extra else 0) for i in range(rows)]
        
        # Create the pattern
        c = np.zeros((N, 2))
        idx = 0
        for row_i in range(rows):
            n_in_row = row_sizes[row_i]
            if n_in_row == 0:
                continue
            # y position
            y = (row_i - (rows - 1) / 2.0) / (rows - 1) * 0.9 if rows > 1 else 0
            # x positions, offset alternating rows
            offset = (0.5 / n_in_row) if (row_i % 2 == 1) else 0
            for j in range(n_in_row):
                x = (j - (n_in_row - 1) / 2.0) / (n_in_row - 1) * 0.9 + offset if n_in_row > 1 else 0
                c[idx] = [x, y]
                idx += 1
        
        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    return best_c if best_c is not None else np.random.uniform(-0.4, 0.4, (N, 2))

def _make_random():
    """Create a random configuration."""
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _local_optimize(c, max_iter=1000, tol=1e-10):
    """Local optimization using gradient-based method on the smooth surrogate."""
    best_c = c.copy()
    best_r = _radius(c)
    
    for it in range(max_iter):
        # Compute the radius and identify constraining pairs
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = d.min()
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = wall_dists.min()
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if r > best_r + tol:
            best_r = r
            best_c = c.copy()
        
        # Find the most constraining pair or wall
        if min_pair_dist / 2.0 < min_wall_dist:
            # Find the pair with minimum distance
            i, j = np.unravel_index(d.argmin(), d.shape)
            # Push apart
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad_i = diff / dist
                grad_j = -diff / dist
            else:
                grad_i = np.array([1.0, 0.0])
                grad_j = -grad_i
            # Move apart slightly
            step = 1e-4
            c[i] += step * grad_i
            c[j] += step * grad_j
        else:
            # Find the point closest to a wall
            flat_idx = wall_dists.argmin()
            i = flat_idx // 2
            k = flat_idx % 2
            # Push away from wall
            if c[i, k] > 0:
                grad = np.zeros(2)
                grad[k] = -1.0
            else:
                grad = np.zeros(2)
                grad[k] = 1.0
            step = 1e-4
            c[i] += step * grad
        
        # Keep inside square
        c = np.clip(c, -0.4999, 0.4999)
    
    return best_c, best_r

def _simulated_annealing(c, time_limit):
    """Simulated annealing with local moves."""
    best_c = c.copy()
    best_r = _radius(c)
    current_c = c.copy()
    current_r = _radius(c)
    
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    
    start_time = time.time()
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Random move: pick a random point and move it
        i = random.randint(0, N - 1)
        move_size = 0.01 * (1 + T / 0.01)
        delta = np.random.normal(0, move_size, 2)
        new_c = current_c.copy()
        new_c[i] += delta
        new_c[i] = np.clip(new_c[i], -0.4999, 0.4999)
        
        new_r = _radius(new_c)
        delta_r = new_r - current_r
        
        if delta_r > 0 or random.random() < np.exp(delta_r / T):
            current_c = new_c
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
        
        T *= alpha
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Start with honeycomb
    c = _make_honeycomb()
    best_c = c.copy()
    best_r = _radius(c)
    
    # Local optimization
    c, r = _local_optimize(c, max_iter=500)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_c
    
    # Simulated annealing
    c, r = _simulated_annealing(best_c, time_limit=min(60.0, deadline - elapsed))
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_c
    
    # Final local optimization
    c, r = _local_optimize(best_c, max_iter=1000)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    return best_c
# EVOLVE-BLOCK-END
