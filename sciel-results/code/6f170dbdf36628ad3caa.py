"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _score_fast(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist). Vectorized."""
    # Boundary distance
    abs_c = np.abs(c)
    r_bound = np.min(0.5 - np.max(abs_c, axis=1))
    
    # Pairwise distances - compute all at once
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf so it's not considered
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement with some randomization."""
    m = 5  # 5x5x5 = 125 points, we use 100
    # Create a 5x5x5 grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add small random perturbation to break symmetry
    grid = grid + np.random.randn(len(grid), 3) * 0.005
    # Clamp to cube
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _clamp_to_cube(c):
    """Clamp centres to be inside the cube [-0.5, 0.5]^3."""
    c = np.clip(c, -0.5, 0.5)
    return c

def _simulated_annealing(c, max_iters=30000, T0=0.01, T1=1e-7, alpha=0.9997, step_scale=0.02, time_limit=90.0):
    """
    Simulated annealing to improve the packing with adaptive step size and multi-point moves.
    """
    c = c.copy()
    current_score = _score_fast(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    n_accepted = 0
    n_total = 0
    start_time = time.time()
    
    # Adaptive step size
    current_step = step_scale
    recent_improvements = 0
    improvement_threshold = 50
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Generate a perturbation: move 1-3 random points
        n_move = 1 if random.random() < 0.6 else (2 if random.random() < 0.8 else 3)
        idxs = random.sample(range(N), n_move)
        
        c_new = c.copy()
        for idx in idxs:
            # Random perturbation with adaptive step size
            delta = np.random.randn(3) * current_step
            c_new[idx] = c_new[idx] + delta
            # Clamp to cube
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _score_fast(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Better or equal, always accept
            c = c_new
            current_score = new_score
            n_accepted += 1
            recent_improvements += 1
        else:
            # Worse, accept with probability exp((new_score - current_score) / T)
            delta_score = new_score - current_score  # negative
            if abs(delta_score) < 1e-10:
                prob = 1.0
            else:
                prob = math.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
                n_accepted += 1
        
        n_total += 1
        
        # Adaptive step size: increase if too many improvements (exploring well), decrease if stuck
        if recent_improvements >= improvement_threshold:
            current_step *= 0.95  # Slightly decrease to refine
            recent_improvements = 0
        elif n_total % 100 == 0 and n_accepted / max(n_total, 1) < 0.1:
            current_step *= 1.1  # Increase to explore more
            current_step = min(current_step, 0.05)  # Cap at reasonable value
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=5000, time_limit=20.0):
    """
    Local refinement: for each pair of closest spheres, try to push them apart.
    Also try to pull spheres away from boundaries if they're the limiting factor.
    """
    c = c.copy()
    current_score = _score_fast(c)
    best_c = c.copy()
    best_score = current_score
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find the limiting constraint
        abs_c = np.abs(c)
        r_bound = np.min(0.5 - np.max(abs_c, axis=1))
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            min_idx = np.argmin(dists)
            i, j = np.unravel_index(min_idx, (N, N))
            
            # Push i and j apart
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff_ij / dist_ij
            
            # Try multiple move sizes
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                
                new_score = _score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        else:
            # Boundary is limiting
            # Find the point and axis that's closest to boundary
            dists_to_faces = 0.5 - abs_c  # shape (N, 3)
            min_val = np.min(dists_to_faces)
            min_idx = np.argmin(dists_to_faces)
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        
        # Random perturbation occasionally to escape local minima
        if it % 50 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.008
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _score_fast(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    start_time = time.time()
    
    # Start with a grid
    c = _initial_grid()
    
    # Run simulated annealing - first pass with larger steps
    c, score = _simulated_annealing(c, max_iters=20000, T0=0.008, T1=1e-6, alpha=0.9996, step_scale=0.02, time_limit=40.0)
    
    # Local refinement
    c, score = _local_refine(c, max_iters=3000, time_limit=15.0)
    
    # Check time
    elapsed = time.time() - start_time
    remaining = 110.0 - elapsed  # Leave 10s buffer
    
    if remaining > 20:
        # Second round of simulated annealing with smaller steps
        c, score2 = _simulated_annealing(c, max_iters=15000, T0=0.004, T1=1e-7, alpha=0.9997, step_scale=0.01, time_limit=min(remaining - 10, 35.0))
        if score2 > score:
            score = score2
            c, score = _local_refine(c, max_iters=2000, time_limit=10.0)
    
    # Final check: ensure all points are strictly inside
    c = np.clip(c, -0.499, 0.499)
    
    return c
# EVOLVE-BLOCK-END
