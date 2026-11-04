import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words and neighbors
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    neighbor_sets = []  # Sets for fast membership testing
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val  # Restore
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
    
    # 2. Greedy Set Cover with vectorized gain computation
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # Build neighbor matrix as a list of arrays for vectorized operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    while np.any(uncovered):
        if time.time() > deadline - 25:
            break
        
        # Vectorized gain computation: for each candidate, sum uncovered in its neighbors
        # We can do this by summing uncovered[neighbor_arrays[i]] for all i
        # But that's still a loop. Let's use a different approach.
        
        # Alternative: For each uncovered point, add 1 to all candidates that cover it
        # This is the transpose of the incidence matrix
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            # All codewords within distance 1 of idx cover idx
            # That's neighbors[idx]
            gains[neighbors[idx]] += 1
        
        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]
        
        if max_gain == 0:
            break
        
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors[best_idx]] = False
        
    # 3. Local Search with enhanced 1-exchange moves
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors[idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if np.min(counts[neighbors[c]]) > 1:
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
            
        # --- Enhanced Swap Pass with multiple strategies ---
        random.shuffle(code)
        
        for c in code:
            if time.time() > deadline - 5:
                break
                
            # Check if c is currently removable
            if np.min(counts[neighbors[c]]) > 1:
                continue
                
            # Get critical points for c: points i in neighbors[c] with counts[i] == 1
            crit_points = [i for i in neighbors[c] if counts[i] == 1]
            
            if not crit_points:
                continue
                
            # Strategy 1: Find replacement covering all critical points
            found_swap = False
            random.shuffle(crit_points)
            
            for p in crit_points:
                if found_swap:
                    break
                    
                cands = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Score: maximize redundancy creation (points with count==2)
                    score = sum(1 for i in neighbors[w] if counts[i] == 2)
                    # Also consider: minimize additional coverage loss
                    # Points that would go from count 1 to 0 if we remove c
                    # But we're adding w, so points covered by both c and w stay covered
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors[best_w]] += 1
                    
                    improved = True
                    found_swap = True
                    break
            
            # Strategy 2: If no swap found, try a "relocation" - move c to a better position
            # even if it doesn't cover all critical points, as long as we can cover them
            # with existing codewords
            if not found_swap:
                # Check if removing c leaves some points uncovered
                # Those must be covered by other codewords
                temp_counts = counts.copy()
                temp_counts[neighbors[c]] -= 1
                
                if np.min(temp_counts) >= 1:
                    # c is redundant after all (shouldn't happen if deletion pass worked)
                    continue
                
                # Points that would become uncovered
                would_be_uncovered = [i for i in neighbors[c] if temp_counts[i] == 0]
                
                if not would_be_uncovered:
                    continue
                
                # For each such point, find a codeword that covers it and isn't c
                # We need to add a codeword that covers ALL would_be_uncovered points
                # OR we can try to replace c with a codeword that covers the critical points
                # while some other codeword covers the rest
                
                # Try to find a single replacement that covers all would_be_uncovered
                # and all crit_points
                all_needed = set(crit_points) | set(would_be_uncovered)
                
                best_w = -1
                best_score = -1
                
                # Candidates must cover all points in all_needed
                # Start with candidates from the first needed point
                first_point = list(all_needed)[0]
                cands = [x for x in neighbors[first_point] if x not in code_set]
                
                for w in cands:
                    valid = True
                    for k in all_needed:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = sum(1 for i in neighbors[w] if counts[i] == 2)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors[best_w]] += 1
                    
                    improved = True

    # Final cleanup: Remove any remaining redundant words
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if np.min(counts[neighbors[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
