import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy for general case
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
    
    # Precompute neighbors as list of numpy arrays for fast slicing
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
        
    # Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        # Vectorized greedy selection is hard with sparse neighbors without heavy memory use.
        # N=6561 is small enough for a loop if optimized.
        best_idx = -1
        best_count = -1
        # Optimization: Only check words that are not already in C? No, we can reuse neighbors.
        # We need to find the word covering the most *uncovered* points.
        # counts[i] = sum(uncovered[neighbors[i]])
        # We can maintain a 'potential' score?
        # For simplicity and speed, we iterate.
        for i in range(N):
            if i in C: # Skip if already chosen? No, neighbors are static.
                # Actually, we don't add to C until we pick.
                pass
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
            
    # Local Search Optimization
    # State: C (list of indices), counts (array of coverage counts)
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        new_code = []
        changed = True
        # We need to be careful with counts. If we remove a word, we decrement counts.
        # If we keep it, counts stay.
        # Iterative removal:
        # If min(counts[neighbors[c]]) >= 2, we can remove c.
        # After removing c, counts for neighbors[c] decrease by 1.
        # This might make other words redundant.
        
        # To avoid modifying counts in place and confusing the logic if we revert,
        # we work on a copy or carefully track.
        # Given the small size, we can just recompute or track carefully.
        # Let's track carefully.
        
        current_code = code
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            improved = True
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        # For each codeword c, find unique points it covers.
        # Try to swap c with c' that covers those unique points.
        
        # To speed up, we can precompute unique points for all c?
        # unique_points[c] = neighbors[c] where counts == 1
        
        # We iterate through C.
        # We need to be careful not to modify C while iterating.
        # We will collect swaps or perform them one by one.
        
        # Optimization: If no reduction happened, we might be stuck.
        # Try swaps.
        
        # Limit swaps per iteration to prevent timeout
        max_swaps = 50
        swaps_done = 0
        
        for c in C:
            if swaps_done >= max_swaps:
                break
            if time.time() - start_time > time_limit:
                break
                
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Should have been removed in reduction
                continue
                
            # If too many unique points, unlikely to find a single c' covering all
            if len(unique_points) > 4:
                continue
                
            # Find candidates c' that cover all unique_points
            # Intersection of neighbors[u] for u in unique_points
            # Start with neighbors of first unique point
            candidates_set = set(neighbors[unique_points[0]].tolist())
            
            for u in unique_points[1:]:
                candidates_set &= set(neighbors[u].tolist())
                if not candidates_set:
                    break
            
            if not candidates_set:
                continue
            
            # Filter candidates: must not be in C (or if in C, it's c itself? No, c is removed)
            # We want c' such that c' not in C \ {c}
            # So c' can be in C only if c' == c (which is a no-op)
            # We want c' not in C at all, or c' == c (but we are swapping, so c' != c)
            # Actually, if c' is in C and c' != c, then we are just moving c to c' position?
            # No, we remove c and add c'. If c' is already in C, we just remove c.
            # But if c' is in C, then c' was covering its own points.
            # If we remove c, and c' was already there, the coverage of c's unique points
            # must be provided by c'.
            # So candidates should be c' such that c' covers unique_points.
            # And we remove c.
            # If c' is in C, we just remove c.
            # If c' is not in C, we remove c and add c'.
            
            # Let's just try all candidates in candidates_set
            for c_prime in candidates_set:
                if c_prime == c:
                    continue
                
                # Check if swap is valid
                # Remove c, Add c_prime (if not already in C)
                
                # We need to check if after removing c, all points covered by c
                # are still covered by someone else (specifically c_prime for the unique ones).
                # The non-unique points of c are covered by >= 1 other word.
                # So if c_prime covers the unique points, the swap is valid.
                
                # But we must ensure c_prime doesn't "uncover" anything?
                # No, adding a word only increases coverage.
                # So the only risk is if we remove c, do we lose coverage?
                # Yes, for the unique points of c.
                # If c_prime covers them, we are safe.
                
                # So the condition is simply: c_prime covers all unique_points of c.
                # We already ensured this by the intersection logic.
                
                # So the swap is valid!
                # Now, does it help?
                # It keeps size same (if c_prime not in C) or reduces size (if c_prime in C).
                # If c_prime in C, we just remove c. Size reduces by 1.
                # If c_prime not in C, size stays same.
                
                # Let's apply the swap and see if it allows further reduction.
                
                # Save state
                old_C = C.copy()
                old_counts = counts.copy()
                
                # Apply swap
                C.remove(c)
                if c_prime not in C:
                    C.append(c_prime)
                
                # Update counts
                counts[neighbors[c]] -= 1
                if c_prime not in old_C:
                    counts[neighbors[c_prime]] += 1
                # Note: if c_prime was in C, we didn't add it, so we don't increment.
                # But we removed c.
                
                # Check validity (should be valid by construction, but just in case)
                if np.any(counts < 1):
                    # Revert
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    # Improvement!
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    swaps_done += 1
                    break # Restart loop over C? Or continue?
                    # Breaking out of candidate loop for this c.
                    # We should probably break out of the for c loop too to restart
                    # because C changed.
                    # But for simplicity, we can just mark improved and continue.
                    # However, C changed, so the current index in 'for c in C' is invalid.
                    # It's safer to break out of the for c loop.
                    
                else:
                    # No reduction.
                    # Do we keep the swap?
                    # Keeping it might help escape local minima.
                    # But it might also waste time.
                    # Let's revert for now to be safe and deterministic.
                    # Or keep it if it's a "neutral" move?
                    # Let's revert to avoid cycling.
                    C = old_C
                    counts = old_counts
                
            if improved:
                # C changed, restart the for c loop
                break
                
    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
