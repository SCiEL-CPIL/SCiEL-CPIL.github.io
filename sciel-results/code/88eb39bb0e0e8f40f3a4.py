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

def _initial_bcc():
    """Create an initial BCC lattice placement."""
    # Use a 4x4x4 grid of BCC unit cells
    # This gives 2 * 4 * 4 * 4 = 128 points. We take the first 100.
    a = 0.25  # Cell size to fit 4 cells in length 1
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner point
                p1 = np.array([i * a - 0.5, j * a - 0.5, k * a - 0.5])
                # Center point
                p2 = np.array([i * a - 0.5 + a/2, j * a - 0.5 + a/2, k * a - 0.5 + a/2])
                points.append(p1)
                points.append(p2)
    
    # Convert to numpy array
    grid = np.array(points)
    
    # We have 128 points. We need 100.
    # To make it more symmetric, we could shuffle, but taking the first 100 is fine.
    # However, the "first 100" might be biased towards one corner.
    # Let's shuffle to ensure we don't have a hole in the middle.
    random.shuffle(points)
    grid = np.array(points[:N])
    
    # Clamp to cube
    grid = np.clip(grid, -0.5, 0.5)
    return grid

def _clamp_to_cube(c):
    """Clamp centres to be inside the cube [-0.5, 0.5]^3."""
    return np.clip(c, -0.5, 0.5)

def _simulated_annealing(c, max_iters=20000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02):
    """
    Simulated annealing to improve the packing.
    Enhanced with targeted moves on the worst pair.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    # Precompute indices for worst pair tracking
    iu = np.triu_indices(N, k=1)
    
    T = T0
    n_accepted = 0
    n_total = 0
    
    for it in range(max_iters):
        # Identify worst pair (min distance)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        min_idx = np.argmin(dists)
        i_worst, j_worst = iu[0][min_idx], iu[1][min_idx]
        
        # Decide which point(s) to move
        # 50% chance: move one of the worst pair
        # 30% chance: move a random point
        # 20% chance: move two random points
        r_move = random.random()
        
        c_new = c.copy()
        
        if r_move < 0.5:
            # Move one of the worst pair
            idx = i_worst if random.random() < 0.5 else j_worst
            delta = np.random.randn(3) * step_scale
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        elif r_move < 0.8:
            # Move a random point
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_scale
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
        else:
            # Move two random points
            idx1, idx2 = random.sample(range(N), 2)
            delta1 = np.random.randn(3) * step_scale
            delta2 = np.random.randn(3) * step_scale
            c_new[idx1] = c_new[idx1] + delta1
            c_new[idx2] = c_new[idx2] + delta2
            c_new[idx1] = np.clip(c_new[idx1], -0.5, 0.5)
            c_new[idx2] = np.clip(c_new[idx2], -0.5, 0.5)
        
        new_score = _score(c_new)
        
        # Acceptance criterion
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            n_accepted += 1
        else:
            delta_score = new_score - current_score
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
    Local refinement: focus on the worst pair.
    """
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, k=1)
    
    for it in range(max_iters):
        # Find the limiting constraint
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        min_idx = np.argmin(dists)
        i, j = iu[0][min_idx], iu[1][min_idx]
        
        r_pair = dists[min_idx] / 2.0
        r_bound = _min_boundary_distance(c)
        
        if r_pair <= r_bound:
            # Pairwise distance is limiting
            # Try to push i and j apart
            vec = c[i] - c[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = vec / dist
            
            # Try multiple step sizes
            improved = False
            for move in [0.005, 0.002, 0.001]:
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
                    improved = True
                    break
            
            if not improved:
                # Try random perturbation of i or j
                idx = i if random.random() < 0.5 else j
                c_new = c.copy()
                c_new[idx] = c_new[idx] + np.random.randn(3) * 0.005
                c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
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
            min_idx_flat = np.argmin(all_dists.flatten())
            i = min_idx_flat // 3
            axis = min_idx_flat % 3
            
            sign = 1 if c[i, axis] > 0 else -1
            improved = False
            for move in [0.005, 0.002, 0.001]:
                c_new = c.copy()
                c_new[i, axis] = np.clip(c_new[i, axis] + sign * move, -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
                    improved = True
                    break
            
            if not improved:
                # Random perturbation
                c_new = c.copy()
                c_new[i] = c_new[i] + np.random.randn(3) * 0.005
                c_new[i] = np.clip(c_new[i], -0.5, 0.5)
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_c = c.copy()
        
        # Occasionally escape local minima
        if it % 50 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] = c_new[idx] + np.random.randn(3) * 0.02
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
    # Start with BCC grid
    c = _initial_bcc()
    
    # Run simulated annealing
    c, score = _simulated_annealing(c, max_iters=20000, T0=0.01, T1=1e-6, alpha=0.9995, step_scale=0.02)
    
    # Local refinement
    c, score = _local_refine(c, max_iters=5000)
    
    # Another round of simulated annealing from the refined state
    c, score2 = _simulated_annealing(c, max_iters=15000, T0=0.005, T1=1e-7, alpha=0.9997, step_scale=0.01)
    if score2 > score:
        score = score2
        c, score = _local_refine(c, max_iters=3000)
    
    return c
# EVOLVE-BLOCK-END
