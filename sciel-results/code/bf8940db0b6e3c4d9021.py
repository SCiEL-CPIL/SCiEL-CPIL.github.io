"""Rewritten program for kissing number in dimension 11."""
import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Generate candidate vectors: all integer vectors with squared norm = 4
    # These are vectors with two ±1 entries and rest 0, OR one ±2 entry and rest 0
    # Actually, for norm^2 = 4: 
    # Type 1: two entries are ±1 (1+1=2? No, 1^2+1^2=2, not 4)
    # Wait: norm^2 = 4 means sum of squares = 4.
    # Options: one entry is ±2 (2^2=4), or two entries with squares summing to 4: 
    #   (±1, ±1) gives 1+1=2, not 4.
    #   (±2, 0) gives 4.
    #   Actually no other integer combinations: 1+1+1+1=4 (four ±1 entries)
    # So candidates with norm^2=4:
    #   - One ±2 entry (D choices * 2 signs = 2D)
    #   - Four ±1 entries (C(D,4) * 2^4)
    # But wait, the task says "minimum pairwise distance is at least the maximum vector norm"
    # If all vectors have norm 2 (norm^2=4), then we need pairwise distance >= 2.
    # distance^2 = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j = 4 + 4 - 2*v_i·v_j = 8 - 2*v_i·v_j
    # We need distance >= 2, so distance^2 >= 4, so 8 - 2*v_i·v_j >= 4, so v_i·v_j <= 2.
    # So we need a set of vectors with norm^2=4 and pairwise dot product <= 2.
    
    # Let's generate all candidates with norm^2 = 4
    candidates = []
    
    # Type 1: one entry is ±2
    for i in range(D):
        for sign in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * sign
            candidates.append(v)
    
    # Type 2: four entries are ±1
    from itertools import combinations
    for combo in combinations(range(D), 4):
        for signs in range(16):
            v = np.zeros(D, dtype=np.int64)
            for bit, idx in enumerate(combo):
                if signs & (1 << bit):
                    v[idx] = 1
                else:
                    v[idx] = -1
            candidates.append(v)
    
    n = len(candidates)
    
    # Build conflict graph: two vectors conflict if their dot product > 2
    # We want maximum independent set in this graph (no conflicts)
    # i.e., max set where all pairwise dot products <= 2
    
    # Compute dot product matrix
    cand_arr = np.array(candidates, dtype=np.int64)
    dot_matrix = cand_arr @ cand_arr.T  # shape (n, n)
    
    # Conflict: dot > 2
    conflict = dot_matrix > 2
    
    # Use greedy + local search to find large independent set
    # Greedy: pick vertices in order of degree (fewest conflicts first)
    
    def greedy_independent_set(conflict, n):
        degrees = conflict.sum(axis=1)
        order = np.argsort(degrees)  # lowest degree first
        selected = set()
        for idx in order:
            if idx in selected:
                continue
            # Check if idx conflicts with any already selected
            if not any(conflict[idx, j] for j in selected):
                selected.add(idx)
        return selected
    
    def local_search(selected, conflict, n, max_iters=500):
        selected = set(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding one vertex
            for i in range(n):
                if i in selected:
                    continue
                if not any(conflict[i, j] for j in selected):
                    selected.add(i)
                    improved = True
            # Try swaps: remove one, add one or two
            sel_list = list(selected)
            for i in sel_list:
                # Try removing i and adding some j not in selected
                new_sel = selected - {i}
                # Try adding one
                for j in range(n):
                    if j in new_sel:
                        continue
                    if not any(conflict[j, k] for k in new_sel):
                        if len(new_sel | {j}) > len(selected):
                            selected = new_sel | {j}
                            improved = True
                            break
                # Try adding two
                if not improved and len(new_sel) + 2 > len(selected):
                    for j in range(n):
                        if j in new_sel or j == i:
                            continue
                        if any(conflict[j, k] for k in new_sel):
                            continue
                        for k in range(j+1, n):
                            if k in new_sel or k == i:
                                continue
                            if conflict[j, k]:
                                continue
                            if any(conflict[k, m] for m in new_sel):
                                continue
                            selected = new_sel | {j, k}
                            improved = True
                            break
                        if improved:
                            break
                if improved:
                    break
        return selected
    
    # Multi-restart greedy + local search
    best_set = set()
    best_size = 0
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        selected = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline:
                break
            if not any(conflict[idx, j] for j in selected):
                selected.add(idx)
        
        # Local search
        selected = local_search(selected, conflict, n, max_iters=100)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Final polish: try to add more
    for i in range(n):
        if i in best_set:
            continue
        if not any(conflict[i, j] for j in best_set):
            best_set.add(i)
    
    # Convert to array
    if len(best_set) == 0:
        # Fallback: return at least one vector
        return np.array(candidates[:1], dtype=np.int64)
    
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
