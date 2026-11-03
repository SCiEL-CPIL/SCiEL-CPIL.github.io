"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _compute_score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    # Vectorized pairwise distance computation
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_pair = np.min(dists) / 2.0
    
    # Boundary distance
    abs_c = np.abs(c)
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
    
    return min(r_pair, r_bound)

def _initial_grid():
    """Create an initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _initial_fcc_like():
    """Create an FCC-like initial placement."""
    # FCC lattice in a cube
    points = []
    n = 5
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i + j + k < N:
                    # FCC: simple cubic + face centers
                    x = (i + 0.5) / n - 0.5
                    y = (j + 0.5) / n - 0.5
                    z = (k + 0.5) / n - 0.5
                    points.append([x, y, z])
                    # Add face-centered points if within bounds
                    if i < n - 1:
                        points.append([x + 0.5/n, y, z])
                    if j < n - 1:
                        points.append([x, y + 0.5/n, z])
                    if k < n - 1:
                        points.append([x, y, z + 0.5/n])
    
    # Take first N points
    points = np.array(points[:N])
    # Scale to fit in cube
    max_coord = np.max(np.abs(points))
    if max_coord > 0.45:
        points = points * (0.45 / max_coord)
    
    return points

def _simulated_annealing(c, max_iters=50000, T0=0.008, T1=1e-8, alpha=0.9995, step_scale=0.02, time_limit=60.0):
    """
    Simulated annealing to improve the packing with adaptive step size.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    start_time = time.time()
    no_improve_count = 0
    patience = 5000
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size based on temperature
        current_step = step_scale * (T / T0) ** 0.5
        
        # Generate a perturbation: move 1-3 random points
        n_move = 1 if random.random() < 0.6 else (2 if random.random() < 0.8 else 3)
        idxs = random.sample(range(N), min(n_move, N))
        
        c_new = c.copy()
        for idx in idxs:
            # Random perturbation with Gaussian distribution
            delta = np.random.randn(3) * current_step
            c_new[idx] = c_new[idx] + delta
            # Clamp to cube
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _compute_score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Better or equal, always accept
            c = c_new
            current_score = new_score
            no_improve_count = 0
        else:
            # Worse, accept with probability exp((new_score - current_score) / T)
            delta_score = new_score - current_score  # negative
            if delta_score > -T * 50:  # Avoid underflow
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
                    no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early termination if no improvement for too long
        if no_improve_count > patience:
            # Reduce temperature more aggressively
            T *= 0.99
            no_improve_count = 0
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=10000, time_limit=30.0):
    """
    Local refinement: iteratively fix the most violating constraint.
    """
    c = c.copy()
    current_score = _compute_score(c)
    best_c = c.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Compute all distances
        iu = np.triu_indices(N, k=1)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        r_pair = np.min(dists) / 2.0
        
        # Boundary distance
        abs_c = np.abs(c)
        dists_x = 0.5 - abs_c[:, 0]
        dists_y = 0.5 - abs_c[:, 1]
        dists_z = 0.5 - abs_c[:, 2]
        r_bound = np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))
        
        improved = False
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            # Push i and j apart
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff_ij / dist_ij
            
            # Try multiple move sizes
            for move in [0.008, 0.004, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        else:
            # Boundary is limiting
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            
            for move in [0.008, 0.004, 0.002, 0.001, 0.0005]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _compute_score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        # Occasionally add random perturbation to escape local minima
        if it % 500 == 0 and it > 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.015
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _compute_score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                improved = True
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Early termination if no improvement
        if not improved and it > 1000:
            break
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid(),
        _initial_random(),
        _initial_fcc_like(),
    ]
    
    best_c = None
    best_score = 0.0
    
    # Total time budget
    total_time_limit = 100.0
    start_total = time.time()
    
    for init_c in initial_configs:
        # Check remaining time
        if time.time() - start_total > total_time_limit - 30:
            break
        
        # Run simulated annealing
        c, score = _simulated_annealing(init_c, max_iters=30000, T0=0.006, T1=1e-8, alpha=0.9996, step_scale=0.018, time_limit=25.0)
        
        # Local refinement
        c, score = _local_refine(c, max_iters=8000, time_limit=15.0)
        
        # Second round of SA from refined state
        c2, score2 = _simulated_annealing(c, max_iters=20000, T0=0.003, T1=1e-9, alpha=0.9997, step_scale=0.01, time_limit=15.0)
        if score2 > score:
            score = score2
            c, score = _local_refine(c2, max_iters=5000, time_limit=10.0)
        
        # Update best
        if score > best_score:
            best_score = score
            best_c = c.copy()
        
        # Check time
        if time.time() - start_total > total_time_limit - 20:
            break
    
    # Final refinement
    if best_c is not None and time.time() - start_total < total_time_limit - 5:
        best_c, _ = _local_refine(best_c, max_iters=3000, time_limit=5.0)
    
    return best_c if best_c is not None else _initial_grid()
# EVOLVE-BLOCK-END
