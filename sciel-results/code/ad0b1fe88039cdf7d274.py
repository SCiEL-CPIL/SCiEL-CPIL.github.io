"""Rewritten program for packing 50 circles in a unit circle using Simulated Annealing."""
import numpy as np
import time
import math

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit

def _compute_score(centers):
    """
    Compute the packing score for given centers.
    Score = min(min_pair_dist / 2, min_bound_dist)
    """
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _get_hex_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) == 0:
        return _get_random_initialization(n, seed=42)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_initialization(n, seed=42)
    
    # Sort by distance to center to pick central ones
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """
    Generate random initialization inside the unit circle.
    """
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.95**2:
                centers[i] = [x, y]
                break
    return centers

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    # Approximate distribution: 1, 6, 12, 12, 9, 10? 
    # Total 50. 
    rings = [
        (6, 0.25),
        (12, 0.5),
        (12, 0.75),
        (10, 0.9)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])

def _simulated_annealing(centers, start_time, time_budget):
    """
    Simulated Annealing to maximize the packing score.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.9995
    # Max perturbation
    max_delta = 0.05
    
    rng = np.random.default_rng(12345)
    
    while time.time() - start_time < time_budget:
        # Perturb a random point
        i = rng.integers(0, N)
        
        # Generate a random move
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0, max_delta)
        
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_point = current[i] + np.array([dx, dy])
        
        # Check if new point is inside the circle (with small margin)
        if np.linalg.norm(new_point) > 0.999:
            continue
            
        # Apply move
        old_point = current[i].copy()
        current[i] = new_point
        
        # Compute new score
        new_score = _compute_score(current)
        
        # Acceptance criterion
        if new_score > current_score or (rng.random() < math.exp((new_score - current_score) / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Revert move
            current[i] = old_point
            
        # Cool down
        T *= alpha
        
        # If T is too small, restart with a slightly higher T to escape local minima
        if T < 0.001:
            T = 0.005
            max_delta = 0.02
            
    return best, best_score

def _polish(centers, start_time, time_budget):
    """
    Local optimization to polish the solution.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.001
    rng = np.random.default_rng(54321)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try moving each point in 4 directions + random
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
                
            # Directions: up, down, left, right, random
            directions = [
                np.array([0, step_size]),
                np.array([0, -step_size]),
                np.array([step_size, 0]),
                np.array([-step_size, 0]),
                rng.normal(0, step_size, 2)
            ]
            
            for d in directions:
                new_point = current[i] + d
                
                if np.linalg.norm(new_point) > 0.999:
                    continue
                    
                old_point = current[i].copy()
                current[i] = new_point
                
                new_score = _compute_score(current)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = current.copy()
                    improved = True
                else:
                    current[i] = old_point
        
        # Reduce step size if no improvement
        if not improved:
            step_size *= 0.9
            if step_size < 1e-5:
                break
                
    return best, best_score

def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Diverse initializations
    initializations = [
        _get_hex_initialization(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
    ]
    
    # Time budget per initialization for SA
    # We have ~115s total. 5 inits.
    # Let's give 15s to SA and 5s to polish per init, but stop if we run out of time.
    # Actually, let's do a fixed number of SA steps or time.
    # Let's try 10s SA + 5s polish per init, but ensure we don't exceed deadline.
    
    for i, init in enumerate(initializations):
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # SA Phase
        sa_time_budget = 10.0
        if time.time() - start_time + sa_time_budget > DEADLINE - 5:
            sa_time_budget = max(0, DEADLINE - 5 - (time.time() - start_time))
            
        if sa_time_budget > 0:
            sa_centers, sa_score = _simulated_annealing(init, start_time, sa_time_budget)
            
            # Polish Phase
            polish_time_budget = 5.0
            if time.time() - start_time + polish_time_budget > DEADLINE - 2:
                polish_time_budget = max(0, DEADLINE - 2 - (time.time() - start_time))
                
            if polish_time_budget > 0:
                final_centers, final_score = _polish(sa_centers, start_time, polish_time_budget)
            else:
                final_centers, final_score = sa_centers, sa_score
                
            if final_score > best_score:
                best_score = final_score
                best_centers = final_centers.copy()
                
    # Final polish on the absolute best if time remains
    if best_centers is not None and time.time() - start_time < DEADLINE - 2:
        final_time_budget = DEADLINE - 2 - (time.time() - start_time)
        if final_time_budget > 0:
            best_centers, best_score = _polish(best_centers, start_time, final_time_budget)
            
    if best_centers is None:
        # Fallback
        best_centers = initializations[0]
        
    return best_centers
