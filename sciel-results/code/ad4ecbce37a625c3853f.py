import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Precompute neighbor sets for fast intersection/membership if needed
    # But we will primarily use array operations for counting
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Greedy: pick word covering most uncovered
    # Optimization: Only check words that might cover uncovered points?
    # For simplicity and speed, scan all. 6561 * ~500 iterations is manageable.
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        # Vectorized sum is faster than loop
        # But we need to find the max. 
        # We can compute counts for all i at once? 
        # counts[i] = sum(uncovered[neighbors[i]])
        # This is equivalent to: for each j, if uncovered[j], add 1 to all i in neighbors[j]
        # Let's try the direct loop first, it's often fast enough in Python if optimized.
        # Actually, let's use a slightly smarter approach:
        # Iterate through uncovered points, increment a counter for their neighbors.
        # This is O(U * 17) where U is number of uncovered.
        # Vs O(N * 17) for scanning all.
        # If U is large, scanning all is similar. If U is small, this is better.
        # Let's stick to the simple scan for now, it's robust.
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
    
    # Build initial counts
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
    
    # Helper to reduce code
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        # Loop until no more removals
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                # Check if all neighbors of c have count >= 2
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, counts)
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # 2. Local Search
    # Strategy:
    # - Identify codewords with "unique" points (points covered only by this codeword).
    # - If a codeword has 0 unique points, it can be removed (handled by reduce).
    # - If a codeword has k unique points (k > 0), we need to swap it with a codeword that covers all k unique points.
    # - We prioritize swapping codewords with small k (easier to find replacement).
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        if len(C) == old_size:
            # No reduction happened. Try to find a swap.
            improved = False
            # Sort codewords by number of unique points to target "easiest" swaps first?
            # Or random shuffle to explore.
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed, but maybe logic missed it?
                    # Force removal
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, unlikely to find a single replacement
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Candidates must be in intersection of N(u) for u in unique_points
                if len(unique_points) == 1:
                    candidates_list = neighbors[unique_points[0]].tolist()
                else:
                    # Intersection of sets is faster for small k
                    # Start with neighbors of first unique point
                    candidates_set = set(neighbors[unique_points[0]].tolist())
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                    if not candidates_set:
                        continue
                    candidates_list = list(candidates_set)
                
                # Filter candidates: must not be in C
                # Using a set for C is faster for membership
                C_set = set(C)
                valid_candidates = [c_prime for c_prime in candidates_list if c_prime not in C_set]
                
                if not valid_candidates:
                    continue
                
                # Heuristic: Pick candidate that covers the most points currently covered by C?
                # Or simply pick one that allows the most removals?
                # Let's pick one randomly from valid candidates to explore space.
                # To be smarter, we can score them:
                # Score = number of points in N(c_prime) that are covered exactly once (potential for removal)
                # Actually, just picking one that covers the unique points is sufficient for validity.
                # The reduction pass will handle the rest.
                c_prime = random.choice(valid_candidates)
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    # Invalid swap
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    # Neutral move: keep it to escape local minimum?
                    # If we keep it, we might get stuck. 
                    # Let's keep it only if it's different from old state (which it is)
                    # and hope future swaps help.
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
        
        if not improved:
            # Try random kicks to escape local minima
            for _ in range(3):
                if time.time() - start_time > time_limit:
                    break
                if not C:
                    break
                c_idx = random.randint(0, len(C)-1)
                c = C[c_idx]
                # Pick a random neighbor of c as the new codeword
                cand = neighbors[c]
                c_prime = cand[random.randint(0, len(cand)-1)]
                if c_prime in C:
                    continue
                
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    break
                else:
                    C = new_C
                    counts = new_counts

    # Final reduction
    final_counts = np.zeros(N, dtype=int)
    for c in best_C:
        final_counts[neighbors[c]] += 1
    
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in best_C:
            if np.min(final_counts[neighbors[c]]) >= 2:
                final_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        best_C = new_C
    
    return [list(all_words[i]) for i in best_C]
