import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    # Parameters for this task are fixed: q=3, n=8, radius=1
    # But we write it generally for the interface.
    
    total_words = q ** n
    
    # Precompute neighbors (ball of radius 1) for all words
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    neighbors = []
    for i in range(total_words):
        word = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        neighbors.append(ball)
        
    # 1. Greedy Set Cover Initialization
    uncovered_mask = np.ones(total_words, dtype=np.int8)
    code = set()
    
    # Optimization: Precompute neighbor indices as numpy arrays for fast sum
    neighbor_arrays = [np.array(neighbors[i]) for i in range(total_words)]
    
    while np.any(uncovered_mask):
        best_c = -1
        best_count = -1
        
        # Find best candidate
        # We can iterate through all, but maybe skip those already in code?
        # Since total_words is 6561, and code size is ~500, it's fine.
        for c in range(total_words):
            if c in code:
                continue
            count = np.sum(uncovered_mask[neighbor_arrays[c]])
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1:
            break
            
        code.add(best_c)
        uncovered_mask[neighbor_arrays[best_c]] = 0
        
    # 2. Local Search: Minimize Code Size
    # State: code (set), cover_count (array)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[neighbor_arrays[c]] += 1
        
    # Helper to update cover count
    def add_to_cover(c):
        cover_count[neighbor_arrays[c]] += 1
        
    def remove_from_cover(c):
        cover_count[neighbor_arrays[c]] -= 1
        
    # Iterative Improvement
    # We try to remove redundant words and swap to enable removals.
    max_iterations = 5000
    for _ in range(max_iterations):
        improved = False
        
        # Phase 1: Remove any word that doesn't uniquely cover any point
        # unique_count[c] = sum(1 for w in neighbors[c] if cover_count[w] == 1)
        # If unique_count[c] == 0, we can remove c.
        
        # We check words in the current code.
        # To speed up, we can check in random order.
        candidates = list(code)
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in code: # Might have been removed in a previous step? No, we don't remove in this loop yet.
                continue
            # Check if c is redundant
            is_redundant = True
            # If any neighbor has count 1, it's not redundant
            # We can break early if we find one.
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                code.remove(c)
                remove_from_cover(c)
                improved = True
                # Restart loop to check for new redundancies
                break # Break to restart the outer loop
        
        if improved:
            continue
            
        # Phase 2: Try Swaps
        # For each c in code, try to swap with c' not in code.
        # A swap is useful if it reduces the number of "unique" points covered by the new set
        # or allows a removal.
        # Heuristic: Swap c with c' if c' covers some points that c uniquely covers,
        # and c' doesn't uniquely cover points that are hard to cover otherwise.
        
        # Simpler heuristic:
        # Calculate "value" of each c in code: number of points it uniquely covers.
        # If value is 0, we would have removed it in Phase 1. So all have value >= 1.
        # We want to find a c with low value and a c' that can cover those points.
        
        # Get list of codewords sorted by unique coverage count
        code_list = list(code)
        
        # Calculate unique coverage for each
        unique_counts = {}
        for c in code_list:
            count = 0
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    count += 1
            unique_counts[c] = count
            
        # Sort by unique count (lowest first)
        sorted_code = sorted(code_list, key=lambda x: unique_counts[x])
        
        swapped = False
        for c_out in sorted_code:
            if swapped:
                break
            # Try to find a c_in
            # c_in should cover the unique points of c_out
            unique_points = [w for w in neighbors[c_out] if cover_count[w] == 1]
            
            if not unique_points:
                continue
                
            # Find c_in not in code that covers max unique_points of c_out
            best_c_in = -1
            best_overlap = 0
            
            # Iterate through all potential c_in? 6561 checks.
            # 6561 * 17 ops = ~100k ops per c_out.
            # If we have 500 c_out, that's 50M ops. Might be slow in Python.
            # Optimization: Only check c_in that are "close" to c_out?
            # Or just sample?
            # Let's try checking all, but break early if we find a perfect match?
            
            for c_in in range(total_words):
                if c_in in code:
                    continue
                    
                # Count how many of c_out's unique points are covered by c_in
                overlap = 0
                for w in unique_points:
                    if w in neighbors[c_in]:
                        overlap += 1
                        if overlap == best_overlap + 1 and overlap == len(unique_points):
                            # Perfect match found, break inner loop?
                            # We still need to check if c_in is good.
                            pass
                            
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_c_in = c_in
                    
                # Early termination if we found a c_in that covers all unique points
                if best_overlap == len(unique_points):
                    # Check if swapping is beneficial
                    # The new c_in might introduce new unique points.
                    # But if it covers all old unique points, the count of unique points for c_out becomes 0.
                    # The new unique points of c_in will be handled in next iterations.
                    break
            
            if best_c_in != -1 and best_overlap > 0:
                # Perform swap
                code.remove(c_out)
                remove_from_cover(c_out)
                code.add(best_c_in)
                add_to_cover(best_c_in)
                swapped = True
                improved = True
                break # Restart loop
        
        if not improved:
            break
            
    # Convert back to list of lists
    result = [to_word(c) for c in code]
    return result
