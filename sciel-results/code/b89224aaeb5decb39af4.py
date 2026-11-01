import numpy as np
import itertools
import time
import heapq

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists for faster neighbor lookups
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # Greedy MIS with min-degree heuristic (initial solution)
    def greedy_mis():
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = np.array([len(adj[i]) for i in range(N4)], dtype=np.int32)
        rng = np.random.default_rng(42)
        
        while np.any(alive) and time.time() < deadline - 20:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected.append(chosen)
            
            # Remove chosen and its neighbors
            to_remove = np.unique(np.concatenate(([chosen], adj[chosen])))
            alive[to_remove] = False
            
            # Update degrees
            rem = np.where(alive)[0]
            if len(rem) > 0:
                for r in rem:
                    degrees[r] -= np.sum(conflicts[r, to_remove])
            
            if time.time() > deadline - 20:
                break
        
        return selected
    
    selected = greedy_mis()
    
    # Local improvement: try to add vectors that conflict with at most 1 selected vector
    # and swap if beneficial
    def local_improve(selected):
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # For each non-selected vector, check if it can be added
            # by removing one conflicting selected vector
            non_selected = [i for i in range(N4) if i not in selected_set]
            # Sort by number of conflicts with selected (fewer first)
            non_selected.sort(key=lambda i: np.sum(conflicts[i, list(selected_set)]))
            
            for i in non_selected[:200]:  # Limit to speed up
                if time.time() > deadline - 10:
                    break
                conflicts_with_selected = [j for j in selected if conflicts[i, j]]
                if len(conflicts_with_selected) == 0:
                    # Can add directly
                    selected.append(i)
                    selected_set.add(i)
                    improved = True
                elif len(conflicts_with_selected) == 1:
                    # Try swapping: remove one, add this
                    j = conflicts_with_selected[0]
                    # Check if i has fewer conflicts with remaining selected
                    remaining_conflicts = sum(1 for k in selected if k != j and conflicts[i, k])
                    # And j's removal doesn't hurt too much
                    # Simple heuristic: accept if i conflicts with fewer than j does with remaining
                    j_remaining_conflicts = sum(1 for k in selected if k != j and conflicts[j, k])
                    if remaining_conflicts < j_remaining_conflicts:
                        selected.remove(j)
                        selected_set.remove(j)
                        selected.append(i)
                        selected_set.add(i)
                        improved = True
        
        return selected
    
    selected = local_improve(selected)
    
    # Try a few different greedy orderings to get diverse solutions
    best_selected = list(selected)
    best_len = len(selected)
    
    for seed in [1, 7, 13, 42, 99]:
        if time.time() > deadline - 15:
            break
        alive = np.ones(N4, dtype=bool)
        sel = []
        degrees = np.array([len(adj[i]) for i in range(N4)], dtype=np.int32)
        rng = np.random.default_rng(seed)
        
        while np.any(alive) and time.time() < deadline - 15:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            sel.append(chosen)
            
            to_remove = np.unique(np.concatenate(([chosen], adj[chosen])))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                for r in rem:
                    degrees[r] -= np.sum(conflicts[r, to_remove])
        
        sel = local_improve(sel)
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine V4 and V1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
