"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _pairwise_distances(c):
    """Compute all pairwise distances for an array of shape (N, 3)."""
    # c has shape (N, 3)
    # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=2))
    # Return upper triangle indices and values
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    return iu, dists

def _min_pairwise_distance(c):
    """Return the minimum pairwise distance."""
    iu, dists = _pairwise_distances(c)
    return np.min(dists)

def _min_boundary_distance(c):
    """Return the minimum distance from any centre to the nearest face of the cube [-0.5, 0.5]^3."""
    # For each point, distance to nearest face is min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # Since cube is symmetric, we can compute per axis: min(0.5 - |coord|)
    abs_c = np.abs(c)
    # Distance to nearest face along each axis
    dists_x = 0.5 - abs_c[:, 0]
    dists_y = 0.5 - abs_c[:, 1]
    dists_z = 0.5 - abs_c[:, 2]
    return np.min(np.minimum.reduce([dists_x, dists_y, dists_z]))

def _score(c):
    """Compute the radius: min(min_pairwise_dist/2, min_boundary_dist)."""
    r_pair = _min_pairwise_distance(c) / 2.0
    r_bound = _min_boundary_distance(c)
    return min(r_pair, r_bound)

def _objective(c):
    """Objective to minimize: negative of the radius."""
    return -_score(c)

def _initial_grid():
    """Create an initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _clamp_to_cube(c):
    """Clamp centres to be inside the cube [-0.5, 0.5]^3."""
    c = np.clip(c, -0.5, 0.5)
    return c

def _simulated_annealing(c, max_iters=20000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02):
    """
    Simulated annealing to improve the packing.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    n_accepted = 0
    n_total = 0
    
    for it in range(max_iters):
        # Generate a perturbation: move one or two random points
        n_move = 1 if random.random() < 0.7 else 2
        idxs = random.sample(range(N), n_move)
        
        c_new = c.copy()
        for idx in idxs:
            # Random perturbation
            delta = np.random.randn(3) * step_scale
            c_new[idx] = c_new[idx] + delta
            # Clamp to cube
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Better or equal, always accept
            c = c_new
            current_score = new_score
            n_accepted += 1
        else:
            # Worse, accept with probability exp((new_score - current_score) / T)
            delta_score = new_score - current_score  # negative
            prob = math.exp(delta_score / T)
            if random.random() < prob:
                c = c_new
                current_score = new_score
                n_accepted += 1
        
        n_total += 1
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Cool down
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=5000):
    """
    Local refinement: for each pair of closest spheres, try to push them apart.
    Also try to pull spheres away from boundaries if they're the limiting factor.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    for it in range(max_iters):
        # Find the limiting constraint
        r_pair = _min_pairwise_distance(c) / 2.0
        r_bound = _min_boundary_distance(c)
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            iu, dists = _pairwise_distances(c)
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            # Push i and j apart
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Coincident, push in random direction
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Move i in direction, j in opposite direction
            move = 0.005
            c_new = c.copy()
            c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
            c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
            
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
            else:
                # Try a smaller move
                move = 0.001
                c_new = c.copy()
                c_new[i] = np.clip(c_new[i] + direction * move, -0.5, 0.5)
                c_new[j] = np.clip(c_new[j] - direction * move, -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
        else:
            # Boundary is limiting
            abs_c = np.abs(c)
            dists_x = 0.5 - abs_c[:, 0]
            dists_y = 0.5 - abs_c[:, 1]
            dists_z = 0.5 - abs_c[:, 2]
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_val = np.min(all_dists)
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            c_new = c.copy()
            c_new[i, axis] = np.clip(c_new[i, axis] + sign * 0.005, -0.5, 0.5)
            
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
            else:
                # Try smaller move
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * 0.001, -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
        
        # Random perturbation occasionally to escape local minima
        if it % 100 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.01
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_score = _score(c_new)
            if new_score > current_score:
                c = c_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_c = c.copy()
    
    return best_c, best_score

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a grid
    c = _initial_grid()
    
    # Run simulated annealing
    c, score = _simulated_annealing(c, max_iters=15000, T0=0.005, T1=1e-7, alpha=0.9997, step_scale=0.015)
    
    # Local refinement
    c, score = _local_refine(c, max_iters=3000)
    
    # Another round of simulated annealing from the refined state
    c, score2 = _simulated_annealing(c, max_iters=10000, T0=0.003, T1=1e-8, alpha=0.9998, step_scale=0.008)
    if score2 > score:
        score = score2
        c, score = _local_refine(c, max_iters=2000)
    
    return c
# EVOLVE-BLOCK-END
