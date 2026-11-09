"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""

import itertools
import time
import random
import numpy as np

N = 16
ALL_POINTS = np.array([(i, j) for i in range(N) for j in range(N)])
ALL_POINTS_SET = set(map(tuple, ALL_POINTS))
NUM_POINTS = len(ALL_POINTS)

# Precompute distance squared matrix
# DIST[i, j] = ||ALL_POINTS[i] - ALL_POINTS[j]||^2
DIST = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    diffs = ALL_POINTS[i+1:] - ALL_POINTS[i]
    DIST[i, i+1:] = np.sum(diffs ** 2, axis=1)
    DIST[i+1:, i] = DIST[i, i+1:]

def _can_add_vectorized(p_idx, chosen_indices):
    """
    Check if point p_idx can be added to set of points chosen_indices.
    chosen_indices is an array of indices.
    Returns True if compatible.
    """
    n = len(chosen_indices)
    if n < 2:
        return True
    
    # Distances from p to chosen points
    d_p = DIST[p_idx, chosen_indices]
    
    # Distances between chosen points
    # d_ij[i, j] = DIST[chosen_indices[i], chosen_indices[j]]
    d_ij = DIST[np.ix_(chosen_indices, chosen_indices)]
    
    # Conditions for isosceles triangle {p, c_i, c_j}:
    # 1. d_p[i] == d_p[j]
    # 2. d_p[i] == d_ij[i, j]
    # 3. d_p[j] == d_ij[i, j]
    
    # Vectorized check
    # d_p_i = d_p[:, None]
    # d_p_j = d_p[None, :]
    
    # We need to check if ANY pair (i, j) with i != j satisfies any condition.
    
    # Mask for i != j
    mask = ~np.eye(n, dtype=bool)
    
    # Cond 1: d_p[i] == d_p[j]
    if np.any((d_p[:, None] == d_p[None, :]) & mask):
        return False
        
    # Cond 2: d_p[i] == d_ij[i, j]
    if np.any((d_p[:, None] == d_ij) & mask):
        return False
        
    # Cond 3: d_p[j] == d_ij[i, j]
    if np.any((d_p[None, :] == d_ij) & mask):
        return False
        
    return True

def _is_maximal_vectorized(chosen_indices):
    """
    Check if set of points chosen_indices is maximal.
    """
    chosen_set = set(chosen_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx in chosen_set:
            continue
        if _can_add_vectorized(p_idx, chosen_indices):
            return False
    return True

def _greedy_random_order(seed):
    """Greedy construction with a random permutation of points."""
    rng = random.Random(seed)
    order = list(range(NUM_POINTS))
    rng.shuffle(order)
    
    chosen_indices = []
    
    for p_idx in order:
        if _can_add_vectorized(p_idx, np.array(chosen_indices)):
            chosen_indices.append(p_idx)
            
    return chosen_indices

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    rng = random.Random(42)
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Greedy restarts
    while time.time() < deadline - 50:
        seed = rng.randint(0, 1000000)
        result = _greedy_random_order(seed)
        if len(result) < best_len:
            best = result
            best_len = len(result)
            
    if best is None:
        best = _greedy_random_order(0)
        best_len = len(best)
        
    # Phase 2: Local Search (Delete and Refill)
    current = list(best)
    
    while time.time() < deadline:
        improved = False
        # Try removing each point
        for i in range(len(current)):
            if time.time() > deadline:
                break
                
            candidate_indices = [x for j, x in enumerate(current) if j != i]
            cand_array = np.array(candidate_indices)
            
            # Check if maximal
            if _is_maximal_vectorized(cand_array):
                # Accept, size decreased
                current = candidate_indices
                best_len = len(current)
                improved = True
                break # Restart loop with new current
                
            else:
                # Not maximal, try to swap
                # Find addable points for candidate
                addable = []
                for p_idx in range(NUM_POINTS):
                    if p_idx in set(candidate_indices):
                        continue
                    if _can_add_vectorized(p_idx, cand_array):
                        addable.append(p_idx)
                        
                if addable:
                    # Try a few swaps
                    rng.shuffle(addable)
                    for p_idx in addable[:10]:
                        new_list = candidate_indices + [p_idx]
                        new_array = np.array(new_list)
                        
                        # Check if new set is maximal
                        if _is_maximal_vectorized(new_array):
                            current = new_list
                            improved = True
                            break
                            
        if not improved:
            # Stuck. Try a random restart.
            if time.time() < deadline - 10:
                seed = rng.randint(0, 1000000)
                result = _greedy_random_order(seed)
                if len(result) < best_len:
                    best = result
                    best_len = len(result)
                    current = list(best)
            else:
                break
                
    # Convert indices back to points
    return [ALL_POINTS[i] for i in best]
