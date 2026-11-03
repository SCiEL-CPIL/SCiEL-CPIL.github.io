"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Generate Candidate Vectors
    # We use weight-4 vectors.
    # Condition: dot(u, v) <= 2 for all u, v in set.
    # For weight 4, norm^2 = 4.
    # If we use weight-2 vectors, norm^2 = 2.
    # Mixed sets are complex. We'll stick to weight-4 for a strong base, 
    # but we will also generate weight-2 as a fallback/seed.
    
    candidates_w4 = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_w4.append(v)
    
    candidates = np.array(candidates_w4, dtype=np.int64)
    n = len(candidates)
    
    # Fallback set: All weight-2 vectors (size 220)
    # They are mutually compatible.
    fallback_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                fallback_vectors.append(v)
    fallback_arr = np.array(fallback_vectors, dtype=np.int64)
    
    # 2. Build Conflict Matrix for Weight-4
    # Dot product matrix
    dots = candidates @ candidates.T
    # Conflict if dot > 2
    incompat = (dots > 2).astype(np.int8)
    np.fill_diagonal(incompat, 0)
    
    # 3. Greedy + Local Search
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Initial Greedy
    degrees = incompat.sum(axis=1)
    order = np.argsort(degrees)
    selected = []
    selected_set = set()
    for i in order:
        if i in selected_set:
            continue
        if not selected:
            selected.append(i)
            selected_set.add(i)
        else:
            # Check compatibility with all selected
            if np.all(incompat[i, np.array(selected)] == 0):
                selected.append(i)
                selected_set.add(i)
                
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
        
    # Random Restarts with Greedy
    # Weights: prefer low degree
    weights = 1.0 / (degrees + 1.0)
    weights /= weights.sum()
    
    max_restarts = 1000
    restart_count = 0
    
    while restart_count < max_restarts:
        if time.time() > deadline - 20.0:
            break
        restart_count += 1
        
        # Random order biased by weights
        # To speed up, we can just shuffle indices and use a simple greedy
        # Or use weighted sampling without replacement
        # For N=5280, full sampling is O(N log N). 
        # Let's try simple shuffle first for speed, then weighted if time permits.
        order = rng.permutation(n)
        
        selected = []
        selected_set = set()
        # Optimization: Keep track of available nodes? 
        # Simple approach: iterate and check.
        for i in order:
            if i in selected_set:
                continue
            if not selected:
                selected.append(i)
                selected_set.add(i)
            else:
                # Vectorized check
                if np.all(incompat[i, np.array(selected)] == 0):
                    selected.append(i)
                    selected_set.add(i)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
        # If we found a very good set, stop early
        if best_size > 1000:
            break

    # Local Search: 1-swap improvement
    # Try to replace one node with another to increase size or enable more additions
    if time.time() < deadline - 10.0 and best_set:
        current_set = list(best_set)
        current_arr = np.array(current_set)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Compute conflicts for all nodes against current set
            if len(current_arr) > 0:
                conflicts = incompat[:, current_arr].sum(axis=1)
            else:
                conflicts = np.zeros(n, dtype=np.int64)
            
            # Mask selected nodes
            in_set = np.zeros(n, dtype=bool)
            in_set[current_arr] = True
            
            # Try to add nodes with 0 conflicts
            zero_conf = np.where((conflicts == 0) & (~in_set))[0]
            if len(zero_conf) > 0:
                # Add all
                for node in zero_conf:
                    current_set.append(node)
                    in_set[node] = True
                current_arr = np.array(current_set)
                improved = True
                continue
                
            # Try swaps: Remove one, add one
            # Heuristic: Remove node with highest "cost" (conflicts with many potential additions?)
            # Simplified: Try removing each node and see if we can add more
            # This is O(N^2) per iteration. Might be slow.
            # Instead, try random swaps.
            
            if len(current_set) > 0:
                # Pick a random node to remove
                rem_idx = rng.integers(0, len(current_set))
                rem_node = current_set[rem_idx]
                
                # Find best node to add
                # Temporarily remove rem_node from conflict calculation?
                # conflicts currently include rem_node.
                # We need conflicts against (current_set - {rem_node}).
                # Let's just recompute for the candidate pool if we find a good one.
                
                # Candidate pool: nodes not in set.
                # We want a node that has low conflict with the rest of the set.
                # Approximation: conflicts[i] - incompat[i, rem_node]
                # Since incompat is 0/1, if incompat[i, rem_node]==1, it reduces conflict by 1.
                
                adjusted_conflicts = conflicts.copy()
                adjusted_conflicts[rem_node] = n # Exclude self
                # Subtract contribution of rem_node
                adjusted_conflicts -= incompat[:, rem_node]
                
                # Mask out current set
                adjusted_conflicts[in_set] = n
                
                # Find best candidate
                best_cand_idx = np.argmin(adjusted_conflicts)
                best_cand_conf = adjusted_conflicts[best_cand_idx]
                
                # If the best candidate has 0 conflicts with the rest, it's a valid swap (size stays same)
                # But we want to increase size. 
                # Actually, if we swap, size stays same. 
                # The "Add all zero conflict" step above handles size increase.
                # So local search should focus on enabling future additions?
                # Or just maximizing set size via greedy restarts is usually sufficient for these problems.
                # Let's stick to the "Add all" step as the primary improvement driver.
                # If no zero-conflict nodes remain, we are at a local maximum for greedy.
                
                # To escape local maxima, we can try removing a random node and re-greedy-ing the rest?
                # That's expensive.
                
                pass

    # Final Result Construction
    if best_size > 0:
        result = candidates[list(best_set)]
        # Verify
        if len(result) > 1:
            # Check min distance
            # dist^2 = 8 - 2*dot
            # min dist^2 >= 4  => max dot <= 2
            dots_res = result @ result.T
            np.fill_diagonal(dots_res, 0)
            if np.max(dots_res) <= 2:
                return result
            else:
                # Invalid, use fallback
                return fallback_arr
    else:
        return fallback_arr
# EVOLVE-BLOCK-END
