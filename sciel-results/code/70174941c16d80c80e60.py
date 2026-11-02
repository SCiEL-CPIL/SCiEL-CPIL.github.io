import numpy as np
import time
import random
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = 145  # 145s out of 150s
    
    # Generate candidate vectors: W1 (norm 1) and W4 (norm 2) vectors
    # W1: single coordinate +/-1
    # W4: two coordinates +/-1
    candidates = []
    
    # W1 vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # W4 vectors
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Greedy construction: start empty, add vectors that don't conflict
    # Conflict: distance < max_norm of current set
    selected = []
    max_norm_sq = 0  # Use squared norms for speed
    
    # Sort candidates by norm (smaller norms first for better packing)
    norms_sq = np.sum(candidates ** 2, axis=1)
    order = np.argsort(norms_sq)
    
    for idx in order:
        if time.time() > deadline:
            break
        v = candidates[idx]
        v_norm_sq = norms_sq[idx]
        
        # Check if adding v would violate constraint
        # Need: min pairwise dist >= max norm
        # If we add v, new max_norm = max(current_max_norm, norm(v))
        # Need all existing vectors to be at distance >= new_max_norm from v
        new_max_norm_sq = max(max_norm_sq, v_norm_sq)
        
        valid = True
        if selected:
            sel_arr = np.array(selected, dtype=np.int64)
            # Compute distances from v to all selected
            diffs = sel_arr - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            if np.min(dists_sq) < new_max_norm_sq:
                valid = False
        
        if valid:
            selected.append(v)
            max_norm_sq = new_max_norm_sq
    
    if not selected:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    best_arr = np.array(selected, dtype=np.int64)
    best_k = len(best_arr)
    
    # Local search: try to add more vectors or swap to improve
    # Try adding vectors that were previously rejected
    rejected = [candidates[i] for i in order if i not in [order[j] for j in range(len(selected))]]
    # This is inefficient; let's use a different approach
    
    # Simulated annealing / local search with add and swap moves
    current = list(selected)
    current_max_norm_sq = max_norm_sq
    
    # Precompute norms of candidates
    cand_norms_sq = norms_sq
    
    def is_valid_config(config):
        """Check if config is valid: min pairwise dist >= max norm"""
        if len(config) <= 1:
            return True
        arr = np.array(config, dtype=np.int64)
        norms = np.sum(arr ** 2, axis=1)
        max_norm_sq_val = np.max(norms)
        # Compute all pairwise distances
        # For efficiency, check incrementally
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                diff = arr[i] - arr[j]
                dist_sq = np.sum(diff ** 2)
                if dist_sq < max_norm_sq_val:
                    return False
        return True
    
    # Try to add more vectors via local search
    # Use a greedy repair approach
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each candidate
        for idx in range(len(candidates)):
            if time.time() > deadline:
                break
            v = candidates[idx]
            v_norm_sq = cand_norms_sq[idx]
            
            # Check if v is already in current
            if any(np.array_equal(v, c) for c in current):
                continue
            
            # Try adding v
            new_config = current + [v]
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            # Check validity
            valid = True
            sel_arr = np.array(current, dtype=np.int64)
            diffs = sel_arr - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            if np.min(dists_sq) < new_max_norm_sq:
                valid = False
            
            if valid:
                current = new_config
                current_max_norm_sq = new_max_norm_sq
                improved = True
                best_k = len(current)
    
    # Try swap moves: replace one vector with another to allow more additions
    # This is complex; for now, return what we have
    final_arr = np.array(current, dtype=np.int64)
    
    # Final verification
    if len(final_arr) > 1:
        norms = np.linalg.norm(final_arr, axis=1)
        max_norm = np.max(norms)
        min_dist = float('inf')
        for i in range(len(final_arr)):
            for j in range(i + 1, len(final_arr)):
                dist = np.linalg.norm(final_arr[i] - final_arr[j])
                if dist < min_dist:
                    min_dist = dist
        
        if min_dist < max_norm - 1e-9:
            # Invalid, return smaller valid set
            # Use just W1 vectors
            fallback = []
            for i in range(D):
                v_pos = [0] * D
                v_neg = [0] * D
                v_pos[i] = 1
                v_neg[i] = -1
                fallback.append(v_pos)
                fallback.append(v_neg)
            return np.array(fallback, dtype=np.int64)
    
    return final_arr
