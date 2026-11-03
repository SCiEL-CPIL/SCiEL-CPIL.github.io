import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    random.seed(42)
    
    # Generate all weight-4 vectors (four ±1 entries)
    # These have norm² = 4, so constraint: u·v ≤ 2
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute dot product matrix (conflict: dot > 2)
    # This is n_cand x n_cand, could be large. n_cand = C(11,4) * 16 = 330 * 16 = 5280
    # 5280^2 = ~28M entries, manageable
    
    # Compute dot products in batches to save memory
    # Actually, let's just compute it all at once
    dots = candidates @ candidates.T
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy: pick vectors that have fewest conflicts
    in_set = np.zeros(n_cand, dtype=bool)
    selected_idx = []
    
    # Initial greedy: pick vector with minimum degree (fewest conflicts)
    degrees = conflict.sum(axis=1)
    
    for _ in range(n_cand):
        if time.time() > deadline - 30:
            break
        
        # Find unselected vector with minimum conflicts to already-selected
        # This is expensive, so let's use a simpler greedy first
        
        # Simple greedy: pick in order of increasing initial degree
        pass
    
    # Better approach: greedy by picking vector with min conflicts to selected set
    # Then local search to improve
    
    # Phase 1: Greedy construction
    in_set = np.zeros(n_cand, dtype=bool)
    selected = []
    
    # Start with a random ordering biased toward low-degree nodes
    order = np.argsort(degrees) + np.random.random(n_cand) * 0.1
    order = np.argsort(order)
    
    for i in order:
        if time.time() > deadline - 60:
            break
        if in_set[i]:
            continue
        # Check if i conflicts with any selected
        if not np.any(conflict[i][in_set]):
            in_set[i] = True
            selected.append(i)
    
    k = len(selected)
    
    # Phase 2: Local search - try to add more or swap
    # Try to add unselected vectors
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        
        improved = False
        unselected = np.where(~in_set)[0]
        if len(unselected) == 0:
            break
        
        # For each unselected, count conflicts with selected
        for i in unselected:
            conflicts_with_selected = np.where(conflict[i][in_set])[0]
            if len(conflicts_with_selected) == 0:
                in_set[i] = True
                selected.append(i)
                k += 1
                improved = True
                break
        
        if not improved:
            # Try swaps: remove one selected, add up to two unselected
            for i in unselected[:100]:
                if time.time() > deadline - 10:
                    break
                conflicts = np.where(conflict[i][in_set])[0]
                if len(conflicts) <= 1:
                    # Remove the conflicting ones, add i
                    for c_idx in conflicts:
                        c_pos = selected.index(c_idx) if c_idx in selected else -1
                        # This is getting complex, let's simplify
                    pass
    
    # Phase 3: Try different initial orderings
    best_k = k
    best_in_set = in_set.copy()
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        in_set = np.zeros(n_cand, dtype=bool)
        selected = []
        
        order = np.random.permutation(n_cand)
        
        for i in order:
            if time.time() > deadline - 10:
                break
            if in_set[i]:
                continue
            if not np.any(conflict[i][in_set]):
                in_set[i] = True
                selected.append(i)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_in_set = in_set.copy()
    
    result = candidates[best_in_set]
    return result.astype(np.int64)
