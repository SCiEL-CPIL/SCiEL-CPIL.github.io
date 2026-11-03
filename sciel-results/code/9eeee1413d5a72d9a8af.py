"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _compute_score_fast(c):
    """Fast computation of the minimum radius."""
    # Compute all pairwise squared distances using broadcasting
    # For N=100, this is 100*100*3 = 30000 operations, very fast
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    # Boundary distance
    abs_c = np.abs(c)
    boundary_dists = 0.5 - np.min(abs_c, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _get_closest_pair(c):
    """Get the indices of the closest pair."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
    return min_idx[0], min_idx[1]

def _get_boundary_violation(c):
    """Get the point and axis with minimum boundary distance."""
    abs_c = np.abs(c)
    boundary_dists = 0.5 - abs_c
    min_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
    return min_idx[0], min_idx[1]

def _initial_grid():
    """Create an initial grid placement with slight randomization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N].copy()
    # Add small random perturbation to break symmetry
    grid += np.random.randn(N, 3) * 0.01
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _simulated_annealing_v2(c, max_iters=30000, T0=0.008, T1=1e-7, alpha=0.9996, 
                              base_step=0.012, time_limit=80.0):
    """
    Improved simulated annealing with adaptive step size and time awareness.
    """
    c = c.copy()
    current_score = _compute_score_fast(c)
    best_c = c.copy()
    best_score = current_score
    
    start_time = time.time()
    T = T0
    step_scale = base_step
    
    for it in range(max_iters):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Adaptive step size: decrease as temperature decreases
        current_step = step_scale * (T / T0) ** 0.3
        current_step = max(current_step, 0.001)
        
        # Generate perturbation: move 1-3 random points
        n_move = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        idxs = random.sample(range(N), n_move)
        
        c_new = c.copy()
        for idx in idxs:
            delta = np.random.randn(3) * current_step
            c_new[idx] = np.clip(c_new[idx] + delta, -0.5, 0.5)
        
        new_score = _compute_score_fast(c_new)
        
        # Acceptance criterion with temperature
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            # Use a modified acceptance probability for better exploration
            prob = math.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _force_based_refine(c, max_iters=10000, time_limit=40.0):
    """
    Force-based local refinement: compute repulsive forces between close pairs
    and attractive forces toward center for boundary-limited points.
    """
    c = c.copy()
    current_score = _compute_score_fast(c)
    best_c = c.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        r_pair = 0.5 * np.min(np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2) 
                                      * (1 - np.eye(N))))
        abs_c = np.abs(c)
        boundary_dists = 0.5 - np.min(abs_c, axis=1)
        r_bound = np.min(boundary_dists)
        
        if r_pair <= r_bound:
            # Pairwise constraint is limiting
            i, j = _get_closest_pair(c)
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple move sizes
            improved = False
            for move in [0.008, 0.004, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _compute_score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                # Try moving a nearby point instead
                # Find points close to i or j
                dists_to_i = np.linalg.norm(c - c[i], axis=1)
                dists_to_j = np.linalg.norm(c - c[j], axis=1)
                close_pts = np.where((dists_to_i < 0.2) | (dists_to_j < 0.2))[0]
                close_pts = close_pts[close_pts != i]
                close_pts = close_pts[close_pts != j]
                if len(close_pts) > 0:
                    k = random.choice(close_pts.tolist())
                    # Move k away from both i and j
                    force = (c[k] - c[i]) + (c[k] - c[j])
                    if np.linalg.norm(force) > 1e-10:
                        force /= np.linalg.norm(force)
                    else:
                        force = np.random.randn(3)
                        force /= np.linalg.norm(force)
                    c_new = c.copy()
                    c_new[k] = np.clip(c_new[k] + force * 0.005, -0.5, 0.5)
                    new_score = _compute_score_fast(c_new)
                    if new_score > current_score:
                        c = c_new
                        current_score = new_score
        else:
            # Boundary constraint is limiting
            i, axis = _get_boundary_violation(c)
            sign = 1 if c[i, axis] > 0 else -1
            
            improved = False
            for move in [0.006, 0.003, 0.0015, 0.0008]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                new_score = _compute_score_fast(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                # Try moving a nearby point
                dists_to_i = np.linalg.norm(c - c[i], axis=1)
                close_pts = np.where(dists_to_i < 0.2)[0]
                close_pts = close_pts[close_pts != i]
                if len(close_pts) > 0:
                    k = random.choice(close_pts.tolist())
                    # Move k away from i
                    diff = c[k] - c[i]
                    if np.linalg.norm(diff) > 1e-10:
                        diff /= np.linalg.norm(diff)
                    else:
                        diff = np.random.randn(3)
                        diff /= np.linalg.norm(diff)
                    c_new = c.copy()
                    c_new[k] = np.clip(c_new[k] + diff * 0.004, -0.5, 0.5)
                    new_score = _compute_score_fast(c_new)
                    if new_score > current_score:
                        c = c_new
                        current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Occasional random kick to escape local minima
        if it % 200 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.008
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _compute_score_fast(c_new)
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
    total_time_limit = 110.0  # Leave 10s buffer
    
    best_c = None
    best_score = 0.0
    
    # Try multiple initial configurations and restarts
    for attempt in range(3):
        if time.time() - start_time > total_time_limit:
            break
        
        # Initialize
        if attempt == 0:
            c = _initial_grid()
        else:
            c = np.random.randn(N, 3) * 0.3
            c = np.clip(c, -0.45, 0.45)
        
        # Simulated annealing
        remaining = total_time_limit - (time.time() - start_time)
        sa_time = min(remaining * 0.5, 60.0)
        c, score = _simulated_annealing_v2(c, max_iters=25000, T0=0.007, T1=1e-7, 
                                            alpha=0.99965, base_step=0.011, 
                                            time_limit=sa_time)
        
        if score > best_score:
            best_score = score
            best_c = c.copy()
        
        # Force-based refinement
        remaining = total_time_limit - (time.time() - start_time)
        fb_time = min(remaining * 0.4, 40.0)
        if fb_time > 5.0:
            c, score = _force_based_refine(c, max_iters=8000, time_limit=fb_time)
            if score > best_score:
                best_score = score
                best_c = c.copy()
        
        # Final fine-tuning SA
        remaining = total_time_limit - (time.time() - start_time)
        if remaining > 10.0:
            c, score = _simulated_annealing_v2(c, max_iters=15000, T0=0.003, T1=1e-8,
                                                alpha=0.9998, base_step=0.006,
                                                time_limit=remaining - 5.0)
            if score > best_score:
                best_score = score
                best_c = c.copy()
            
            # Final refinement
            remaining = total_time_limit - (time.time() - start_time)
            if remaining > 5.0:
                c, score = _force_based_refine(c, max_iters=3000, time_limit=remaining - 2.0)
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
    
    # Final check
    if best_c is None:
        best_c = _initial_grid()
    
    return best_c
# EVOLVE-BLOCK-END
