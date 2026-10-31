import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # Generate candidates with squared norm 4
    # Type 1: 4 non-zero entries, each +/- 1
    candidates = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates.append(v)
            
    # Type 2: 1 non-zero entry, value +/- 2
    for i in range(D):
        candidates.append(np.roll(np.array([2,0,0,0,0,0,0,0,0,0,0]), i))
        candidates.append(np.roll(np.array([-2,0,0,0,0,0,0,0,0,0,0]), i))
        
    candidates = np.array(candidates)
    n = len(candidates)
    
    # Build conflict graph: edge (i,j) exists if vectors i and j conflict (dot > 2)
    # We want an independent set in this graph (set of mutually non-conflicting vectors)
    
    # Compute all pairwise dot products
    # For memory efficiency, process in blocks if needed
    # n is about 5280, so n^2 is ~28M, which is manageable
    dots = candidates @ candidates.T
    
    # Conflict matrix: True where dot > 2 (excluding diagonal)
    np.fill_diagonal(dots, 0)
    conflict = dots > 2
    np.fill_diagonal(conflict, False)
    
    # Build adjacency matrix for the compatibility graph (non-conflict)
    # compatible[i,j] = True if vectors i and j can coexist
    compatible = ~conflict
    np.fill_diagonal(compatible, False)
    
    # Spectral approach: use Fiedler vector to find dense subgraph
    # Convert compatible to weighted graph and compute Laplacian
    # For large graphs, use sparse methods or approximation
    
    # Simpler approach: use greedy with spectral ordering
    # Compute degrees in compatibility graph
    degrees = compatible.sum(axis=1)
    
    # Try multiple strategies
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy by degree (highest degree first)
    if time.time() < deadline - 10:
        order = np.argsort(-degrees)
        selected = []
        for idx in order:
            if time.time() > deadline - 5:
                break
            if len(selected) == 0:
                selected.append(idx)
                continue
            # Check compatibility
            sel_vecs = candidates[selected]
            dots_sel = sel_vecs @ candidates[idx]
            if np.max(dots_sel) <= 2:
                selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Strategy 2: Randomized greedy with multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        perm = np.random.permutation(n)
        selected = []
        
        for idx in perm:
            if time.time() > deadline - 5:
                break
            if len(selected) == 0:
                selected.append(idx)
                continue
            
            sel_vecs = candidates[selected]
            dots_sel = sel_vecs @ candidates[idx]
            if np.max(dots_sel) <= 2:
                selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Strategy 3: Local search / simulated annealing refinement
    if best_set and time.time() < deadline - 20:
        selected_set = set(best_set)
        current_size = len(selected_set)
        
        # Try to improve by swapping or adding
        # For each non-selected vector, check if it can be added
        # For each selected vector, check if removing it allows adding more
        
        # Simple local search: try to add vectors that fit
        added_any = True
        while added_any and time.time() < deadline - 10:
            added_any = False
            for idx in range(n):
                if time.time() > deadline - 10:
                    break
                if idx in selected_set:
                    continue
                
                # Check if idx can be added
                sel_list = list(selected_set)
                if len(sel_list) == 0:
                    selected_set.add(idx)
                    added_any = True
                    continue
                
                sel_vecs = candidates[sel_list]
                dots_sel = sel_vecs @ candidates[idx]
                if np.max(dots_sel) <= 2:
                    selected_set.add(idx)
                    added_any = True
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = list(selected_set)
    
    # Convert to result array
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback
        result = np.array([[1,1,0,0,0,0,0,0,0,0,0]])
    
    return result
