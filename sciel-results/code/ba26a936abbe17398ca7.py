"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 200

def _compute_radius(pos):
    """Compute the packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    d = cdist(pos, pos)
    # Set diagonal to inf
    np.fill_diagonal(d, np.inf)
    min_pair = float(np.min(d))
    
    return min(min_pair / 2.0, min_boundary)

def _initial_hex_grid(n, jitter=0.0):
    """Create initial hexagonal lattice packing."""
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col
            y = row * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n], dtype=np.float64)
    
    # Center
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.49, 0.49]
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.49 / max_coord
        points *= scale
    
    if jitter > 0:
        rng = np.random.RandomState(42)
        points += rng.uniform(-jitter, jitter, size=points.shape)
        points = np.clip(points, -0.49, 0.49)
    
    return points

def _initial_random(n, seed=0):
    """Create random initial positions."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-0.49, 0.49, size=(n, 2))
    return points

def _force_directed_optimize(pos, max_iter=5000, lr_init=0.001, lr_end=0.00001, 
                              boundary_weight=1.0, time_limit=60.0):
    """
    Force-directed optimization:
    - Repulsive force between all pairs (inverse square law)
    - Attractive force toward center to keep points in bounds
    - Boundary penalty to keep points inside the square
    
    Uses adaptive learning rate with cooling.
    """
    start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    best_pos = pos.copy()
    best_r = _compute_radius(pos)
    
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        diff = pos[i_idx] - pos[j_idx]
        dist_sq = np.sum(diff**2, axis=1)
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: F = k / d^2 * direction
        # Use a softening to avoid singularity
        k = 1.0
        force_mag = k / (dist_sq + 0.01)
        force_vec = force_mag[:, np.newaxis] * diff / dist[:, np.newaxis]
        
        # Apply forces to positions
        forces = np.zeros_like(pos)
        np.add.at(forces, i_idx, force_vec)
        np.add.at(forces, j_idx, -force_vec)
        
        # Boundary forces: push points away from boundaries
        # x component
        x = pos[:, 0]
        y = pos[:, 1]
        
        # Right boundary (x > 0)
        dist_right = 0.5 - x
        force_right = np.where(dist_right < 0.1, (0.1 - dist_right) * 10.0, 0.0)
        # Left boundary (x < 0)
        dist_left = 0.5 + x
        force_left = np.where(dist_left < 0.1, (0.1 - dist_left) * 10.0, 0.0)
        
        # Top boundary (y > 0)
        dist_top = 0.5 - y
        force_top = np.where(dist_top < 0.1, (0.1 - dist_top) * 10.0, 0.0)
        # Bottom boundary (y < 0)
        dist_bottom = 0.5 + y
        force_bottom = np.where(dist_bottom < 0.1, (0.1 - dist_bottom) * 10.0, 0.0)
        
        forces[:, 0] += -force_right + force_left
        forces[:, 1] += -force_top + force_bottom
        
        # Centering force to prevent drift
        forces -= 0.01 * pos
        
        # Adaptive learning rate
        progress = it / max_iter
        lr = lr_init * (lr_end / lr_init) ** progress
        
        # Update positions
        pos += lr * forces
        pos = np.clip(pos, -0.499, 0.499)
        
        # Evaluate
        r = _compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # Early stopping if no improvement for a while
        if it % 500 == 0 and it > 0:
            # Check if we've stalled
            pass
    
    return best_pos, best_r

def _simulated_annealing(pos, max_iter=20000, t_init=0.05, t_end=0.001, time_limit=60.0):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    
    best_pos = pos.copy()
    best_r = _compute_radius(pos)
    current_r = best_r
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature
        progress = it / max_iter
        temp = t_init * (t_end / t_init) ** progress
        
        # Pick a random point
        i = np.random.randint(n)
        
        # Random perturbation
        delta = np.random.randn(2) * temp
        new_pos = pos.copy()
        new_pos[i] += delta
        new_pos = np.clip(new_pos, -0.499, 0.499)
        
        new_r = _compute_radius(new_pos)
        
        # Accept or reject
        if new_r >= current_r:
            pos = new_pos
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
        else:
            # Boltzmann acceptance
            delta_r = new_r - current_r
            if np.random.rand() < np.exp(delta_r / (temp + 1e-10)):
                pos = new_pos
                current_r = new_r
    
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # Strategy 1: Hex grid + force-directed optimization
    hex_pos = _initial_hex_grid(N, jitter=0.02)
    pos1, r1 = _force_directed_optimize(hex_pos, max_iter=3000, time_limit=30.0)
    if r1 > best_r:
        best_r = r1
        best_pos = pos1.copy()
    
    if time.time() > deadline - 5:
        return best_pos
    
    # Strategy 2: Simulated annealing from best
    pos2, r2 = _simulated_annealing(best_pos, max_iter=10000, t_init=0.03, t_end=0.0005, time_limit=40.0)
    if r2 > best_r:
        best_r = r2
        best_pos = pos2.copy()
    
    if time.time() > deadline - 5:
        return best_pos
    
    # Strategy 3: Force-directed from random start
    rand_pos = _initial_random(N, seed=123)
    pos3, r3 = _force_directed_optimize(rand_pos, max_iter=3000, time_limit=20.0)
    if r3 > best_r:
        best_r = r3
        best_pos = pos3.copy()
    
    if time.time() > deadline - 5:
        return best_pos
    
    # Strategy 4: Simulated annealing from best
    pos4, r4 = _simulated_annealing(best_pos, max_iter=8000, t_init=0.02, t_end=0.0003, time_limit=30.0)
    if r4 > best_r:
        best_r = r4
        best_pos = pos4.copy()
    
    if time.time() > deadline - 3:
        return best_pos
    
    # Strategy 5: Local refinement - small perturbations of best
    pos5 = best_pos.copy()
    current_r = best_r
    for _ in range(200):
        if time.time() > deadline - 2:
            break
        i = np.random.randint(N)
        delta = np.random.randn(2) * 0.005
        new_pos = pos5.copy()
        new_pos[i] += delta
        new_pos = np.clip(new_pos, -0.499, 0.499)
        new_r = _compute_radius(new_pos)
        if new_r > current_r:
            pos5 = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pos = pos5.copy()
    
    return best_pos
