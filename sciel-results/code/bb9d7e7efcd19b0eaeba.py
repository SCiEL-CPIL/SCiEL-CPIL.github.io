# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _get_bottlenecks(c):
    """
    Returns (min_pair_dist, min_boundary_dist, idx_i, idx_j, idx_k)
    where (i,j) is the closest pair and k is the closest to boundary.
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_idx = np.argmin(dists)
        # Convert triu index to (i, j)
        # scipy pdist uses condensed form. 
        # For N=100, we can just use cdist for simplicity in finding indices if N is small enough, 
        # but pdist is faster. Let's stick to pdist and decode index.
        # Decoding triu index is annoying. Let's use cdist for index finding, it's O(N^2) but N=100 is small.
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair_dist = np.min(D)
        idx_i, idx_j = np.unravel_index(np.argmin(D), D.shape)
    else:
        min_pair_dist = np.inf
        idx_i = idx_j = -1
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    idx_k = np.argmin(boundary_dists)
    
    return min_pair_dist, min_boundary_dist, idx_i, idx_j, idx_k

def _relax(c, time_limit=5.0):
    """
    Fast relaxation to remove overlaps.
    Pushes spheres apart if they overlap.
    """
    start_time = time.time()
    c = c.copy()
    # We want to increase min distance. 
    # Simple iterative push-apart.
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_d = np.min(D)
        if min_d > 0.01: # If min dist is decent, stop? No, we want to maximize, but this is just init.
             # Actually, let's just run a few iterations to get a valid config.
             pass
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if i == j: continue
        # Push i away from j and j away from i
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            diff = np.random.randn(3)
            dist = np.linalg.norm(diff)
        unit = diff / dist
        # Move them apart by a fraction of the overlap? 
        # Just move by a small step in opposite directions
        step = 0.01
        c[i] += unit * step
        c[j] -= unit * step
        # Clamp
        c[i] = np.clip(c[i], -CUBE_HALF, CUBE_HALF)
        c[j] = np.clip(c[j], -CUBE_HALF, CUBE_HALF)
        
    return c

def _sa_conflict_driven(c, time_limit=30.0):
    """
    Simulated Annealing that targets the bottleneck constraints.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = 0.01
    T_min = 1e-5
    alpha = 0.999
    step_size = 0.02
    
    # Probability of targeting the bottleneck vs random move
    p_target = 0.7
    
    while time.time() - start_time < time_limit:
        c_new = c.copy()
        
        # Get current bottlenecks
        min_pair_dist, min_boundary_dist, i, j, k = _get_bottlenecks(c)
        
        r = random.random()
        
        if r < p_target:
            # Targeted move
            # Decide if pair or boundary is the limiting factor
            if min_pair_dist / 2.0 < min_boundary_dist:
                # Pair is limiting
                # Move i away from j
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    diff = np.random.randn(3)
                    dist = np.linalg.norm(diff)
                unit = diff / dist
                # Add noise to direction to escape local minima
                noise = np.random.randn(3) * 0.5
                dir_vec = unit + noise
                dir_vec /= np.linalg.norm(dir_vec)
                
                c_new[i] += dir_vec * step_size
                # Optionally move j in opposite direction? 
                # Moving one is enough to increase dist if it's the only constraint.
                # But moving both might be better for global structure? 
                # Let's just move i for now to keep it simple and fast.
            else:
                # Boundary is limiting
                # Move k away from the nearest face
                # Find which face is closest
                pos = c[k]
                # dist to -0.5 x: pos[0] + 0.5
                # dist to +0.5 x: 0.5 - pos[0]
                # ...
                # The point k is closest to the face corresponding to the max abs component
                # Actually, boundary_dists = 0.5 - max(abs(c)).
                # So the limiting face is the one where |c_k_dim| is max.
                max_abs_idx = np.argmax(np.abs(pos))
                sign = 1 if pos[max_abs_idx] > 0 else -1
                # Move inwards
                dir_vec = np.zeros(3)
                dir_vec[max_abs_idx] = -sign
                # Add noise
                noise = np.random.randn(3) * 0.5
                dir_vec = dir_vec + noise
                dir_vec /= np.linalg.norm(dir_vec)
                
                c_new[k] += dir_vec * step_size
        else:
            # Random move
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c_new[idx] += delta
        
        # Clamp
        c_new = np.clip(c_new, -CUBE_HALF, CUBE_HALF)
        
        new_score = _score(c_new)
        
        # Acceptance
        if new_score > current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_score = new_score - current_score
            # Avoid overflow/underflow issues in exp
            if delta_score > -100 * T: # exp(-100) is very small
                prob = np.exp(delta_score / T)
                if random.random() < prob:
                    c = c_new
                    current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if random.random() < 0.1:
                T = 0.005
                step_size = 0.01

        # Adaptive step size
        # Reduce step size as we cool
        step_size = max(0.005, 0.02 * (T / 0.01))

    return best_c, best_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total.
    # 1. Initialization and Relaxation
    # Use a few random starts, relax them, then SA.
    
    time_limit_per_run = 25.0
    num_runs = 4 # 4 * 25 = 100s, leaving 20s for overhead/final polish
    
    for i in range(num_runs):
        # Random start
        c_init = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))
        
        # Relax to get a valid, non-overlapping configuration
        c_relaxed = _relax(c_init, time_limit=2.0)
        
        # Run Conflict-Driven SA
        c_opt, s_opt = _sa_conflict_driven(c_relaxed, time_limit=time_limit_per_run)
        
        if s_opt > best_score:
            best_score = s_opt
            best_c = c_opt
            
    # Final Polish: Run a short SA with lower T on the best solution
    if best_c is not None:
        c_polished, s_polished = _sa_conflict_driven(best_c, time_limit=15.0)
        if s_polished > best_score:
            best_score = s_polished
            best_c = c_polished

    return best_c
# EVOLVE-BLOCK-END
