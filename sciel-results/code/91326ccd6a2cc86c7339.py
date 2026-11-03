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
    abs_c = np.abs(c)
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
    """Create an initial grid placement with better spacing."""
    m = 5  # 5x5x4 = 100 points
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, m)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, m)
    zs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, 4)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _clamp_to_cube(c):
    """Clamp centres to be inside the cube [-0.5, 0.5]^3."""
    c = np.clip(c, -0.5, 0.5)
    return c

def _simulated_annealing(c, max_iters=20000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02):
    """
    Simulated annealing to improve the packing with vectorized operations.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T0
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        # Generate a perturbation: move one or two random points
        n_move = 1 if rng.random() < 0.7 else 2
        idxs = rng.choice(N, size=n_move, replace=False)
        
        c_new = c.copy()
        for idx in idxs:
            delta = rng.standard_normal(3) * step_scale
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
        else:
            delta_score = new_score - current_score
            if delta_score > -100 * T:  # Avoid underflow
                prob = math.exp(delta_score / T)
                if rng.random() < prob:
                    c = c_new
                    current_score = new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
        
        # Adaptive cooling
        T *= alpha
        
        if T < T1:
            break
    
    return best_c, best_score

def _local_refine(c, max_iters=5000):
    """
    Local refinement: iteratively fix the most constraining pairs and boundaries.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    rng = np.random.default_rng(123)
    
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
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Try multiple step sizes
            for move in [0.01, 0.005, 0.002, 0.001]:
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
                    break
        else:
            # Boundary is limiting
            abs_c = np.abs(c)
            dists_x = 0.5 - abs_c[:, 0]
            dists_y = 0.5 - abs_c[:, 1]
            dists_z = 0.5 - abs_c[:, 2]
            all_dists = np.column_stack([dists_x, dists_y, dists_z])
            min_idx = np.argmin(all_dists.flatten())
            i = min_idx // 3
            axis = min_idx % 3
            
            # Push point i away from the boundary
            sign = 1 if c[i, axis] > 0 else -1
            
            for move in [0.01, 0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    break
        
        # Random perturbation occasionally to escape local minima
        if it % 50 == 0:
            n_perturb = rng.integers(1, 4)
            for _ in range(n_perturb):
                idx = rng.integers(0, N)
                c_new = c.copy()
                c_new[idx] = c_new[idx] + rng.standard_normal(3) * 0.015
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
    
    # Run simulated annealing with multiple restarts
    best_c = c
    best_score = _score(c)
    
    for restart in range(3):
        # Perturb the current best
        if restart > 0:
            c = best_c + np.random.randn(N, 3) * 0.05
            c = np.clip(c, -0.5, 0.5)
        else:
            c = _initial_grid()
        
        # Run simulated annealing
        c, score = _simulated_annealing(c, max_iters=12000, T0=0.008, T1=1e-7, alpha=0.9996, step_scale=0.02)
        
        # Local refinement
        c, score = _local_refine(c, max_iters=4000)
        
        if score > best_score:
            best_score = score
            best_c = c.copy()
    
    return best_c
# EVOLVE-BLOCK-END
