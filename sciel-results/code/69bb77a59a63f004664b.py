import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate all integer vectors in R^11 with squared norm 4
    # Type A: one entry is ±2, rest 0 -> 22 vectors
    # Type B: four entries are ±1, rest 0 -> C(11,4)*16 = 330*16 = 5280 vectors
    # Total: 5302
    
    candidates = []
    
    # Type A: (±2, 0, ..., 0)
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B: four ±1 entries
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)  # 5302
    
    # Compute dot product matrix
    M = candidates @ candidates.T  # 5302 x 5302
    
    # Incompatibility: dot > 2 (for i != j)
    # We need a set S such that for all i,j in S, i!=j: M[i,j] <= 2
    # Incompatibility matrix: inc[i,j] = (M[i,j] > 2) and i != j
    inc = (M > 2).astype(np.int8)
    np.fill_diagonal(inc, 0)
    
    # Greedy with multiple strategies
    best_set = None
    best_size = 0
    
    def time_left():
        return deadline - time.time()
    
    # Strategy 1: Random orderings
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time_left() < 5:
            break
        order = rng.permutation(N)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx is compatible with all selected
            if not selected:
                selected.append(idx)
                selected_set.add(idx)
                continue
            # Check dot products with all selected
            # M[idx, selected] <= 2 for all
            dots = M[idx, selected]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search: try to improve by adding/removing
    if best_set is not None:
        selected_arr = np.array(best_set)
        selected_bool = np.zeros(N, dtype=bool)
        selected_bool[selected_arr] = True
        
        # Precompute for each vector, which selected vectors it conflicts with
        # conflict_count[i] = number of selected vectors j where M[i,j] > 2
        # conflict_mask[i] = boolean mask of selected vectors that conflict with i
        
        # For local search, we'll try:
        # 1. Add a random unselected vector if compatible
        # 2. If not compatible, try swapping: remove one conflicting, add new if it improves
        
        unselected = np.where(~selected_bool)[0]
        
        for ls_iter in range(50000):
            if time_left() < 3:
                break
            
            if len(unselected) == 0:
                break
                
            # Pick a random unselected vector
            idx = unselected[np.random.randint(len(unselected))]
            
            # Check conflicts with selected
            conflicts = np.where(selected_bool & (M[idx] > 2))[0]
            
            if len(conflicts) == 0:
                # Can add directly
                selected_bool[idx] = True
                unselected = unselected[unselected != idx]
                selected_arr = np.concatenate([selected_arr, [idx]])
                best_size = len(selected_arr)
            else:
                # Try swapping: remove one conflict, add idx
                # Only beneficial if we can add idx and the removed vector wasn't needed
                # Simple heuristic: try removing the conflict that has the most other conflicts
                # For now, just try random swap
                if np.random.random() < 0.3:
                    rem_idx = conflicts[np.random.randint(len(conflicts))]
                    # Remove rem_idx, add idx
                    selected_bool[rem_idx] = False
                    selected_bool[idx] = True
                    # Update unselected
                    unselected = np.where(~selected_bool)[0]
                    # Rebuild selected_arr
                    selected_arr = np.where(selected_bool)[0]
                    best_size = len(selected_arr)
    
    # Final greedy pass on remaining unselected
    if best_set is not None and time_left() > 5:
        selected_bool = np.zeros(N, dtype=bool)
        selected_bool[best_set] = True
        unselected = np.where(~selected_bool)[0]
        
        # Try to add more
        for idx in unselected:
            if time_left() < 3:
                break
            if np.all(M[idx, selected_bool] <= 2):
                selected_bool[idx] = True
                unselected = unselected[unselected != idx]
        
        best_set = np.where(selected_bool)[0].tolist()
        best_size = len(best_set)
    
    if best_set is None:
        # Fallback
        vectors = candidates[:22]  # Just the Type A vectors
        return vectors.astype(np.int64)
    
    return candidates[best_set].astype(np.int64)
