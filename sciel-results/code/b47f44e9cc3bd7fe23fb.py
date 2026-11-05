import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with radius 'radius'.
    Uses a greedy removal algorithm starting from a direct product construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate the base code using direct product of tetracode and identity
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Base code size: 9 * 3^(n-4)
    # For n=6, size is 81.
    base_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            base_code.append(h + list(tail))
    
    # Convert to list of tuples for hashability
    current_code = [tuple(c) for c in base_code]
    
    # 2. Precompute coverage for all possible words in the space?
    # Space size is q^n. For q=3, n=6, size is 729.
    # It is efficient to precompute the neighbors for each word in the space.
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance 'radius' of all_words[i]
    # Actually, we need to know which codewords cover which space words.
    # Let's represent coverage as a boolean matrix or list of sets.
    # Given small size (729 words), a list of sets of covered word-indices is fine.
    
    # For each word in the space, find its neighbors
    space_neighbors = []
    for w in all_words:
        w_t = tuple(w)
        neighs = [w_t]
        # Generate all words at distance 1
        for i in range(n):
            for v in range(q):
                if v == w_t[i]:
                    continue
                new_w = list(w_t)
                new_w[i] = v
                neighs.append(tuple(new_w))
        # If radius > 1, we would need more, but task is radius 1 typically. 
        # The prompt says "radius" is passed. 
        # For radius 1, the above is sufficient. 
        # If radius > 1, we'd need to expand. 
        # Let's assume radius=1 for efficiency, but handle general case if time permits?
        # The task contract says "checked over the whole space".
        # For q=3, n=6, radius=1 is the standard case.
        # Let's stick to radius=1 logic for speed, as radius>1 is much easier to cover.
        space_neighbors.append(neighs)
    
    # Initialize coverage sets
    # coverage[c] is the set of word indices in the space covered by codeword c
    coverage = []
    for c in current_code:
        covered = set()
        # Find all words within distance 'radius' of c
        # Efficient way: iterate all words? No, 729 is small.
        # Or generate neighbors of c.
        # Let's generate neighbors of c.
        c_list = list(c)
        # Distance 0
        covered.add(word_to_idx[c])
        # Distance 1
        for i in range(n):
            orig = c_list[i]
            for v in range(q):
                if v == orig:
                    continue
                new_c = c_list[:]
                new_c[i] = v
                nc = tuple(new_c)
                if nc in word_to_idx:
                    covered.add(word_to_idx[nc])
        # If radius > 1, we need more. 
        # For now, assume radius=1. If radius is larger, the greedy might be less effective 
        # but the initial code is still valid. 
        # To be safe and general, let's check distance for all words if radius > 1? 
        # No, 729 * 81 checks is ~60k, which is fast.
        # Let's just do it properly for any radius.
        if radius > 1:
            covered = set()
            for w_idx, w in enumerate(all_words):
                dist = sum(1 for a, b in zip(c, w) if a != b)
                if dist <= radius:
                    covered.add(w_idx)
        
        coverage.append(covered)
    
    # 3. Greedy Removal Loop
    # We want to remove codewords that are "redundant".
    # A codeword is redundant if every word it covers is also covered by at least one other codeword.
    
    # To make this efficient, we can track the number of codewords covering each word in the space.
    # cover_count[w_idx] = number of codewords in current_code that cover w_idx.
    cover_count = np.zeros(len(all_words), dtype=int)
    for c_idx, c in enumerate(current_code):
        for w_idx in coverage[c_idx]:
            cover_count[w_idx] += 1
            
    # active_code: list of indices of current_code that are still active
    active_code = list(range(len(current_code)))
    
    # We will repeatedly find a redundant codeword and remove it.
    # To find a redundant codeword, we can check each active codeword.
    # A codeword c is redundant if for all w in coverage[c], cover_count[w] > 1.
    
    # Optimization: 
    # Instead of checking all active codewords every time, we can prioritize.
    # But with ~80 codewords, checking all is cheap.
    
    removed = 0
    while active_code:
        if time.time() > deadline:
            break
            
        # Find a candidate to remove
        candidate_idx = -1
        # Heuristic: Remove the one with the smallest "criticality"?
        # Criticality = number of words it covers exclusively (cover_count == 1).
        # If criticality is 0, it is safe to remove.
        
        min_criticality = float('inf')
        best_candidate = -1
        
        for c_idx in active_code:
            # Count how many words in coverage[c_idx] have cover_count == 1
            crit = 0
            for w_idx in coverage[c_idx]:
                if cover_count[w_idx] == 1:
                    crit += 1
                    # Early exit if we already have more than current best
                    if crit >= min_criticality:
                        break
            
            if crit == 0:
                # Found a perfectly redundant word, remove immediately
                best_candidate = c_idx
                min_criticality = 0
                break
            elif crit < min_criticality:
                min_criticality = crit
                best_candidate = c_idx
        
        if best_candidate == -1:
            # No redundant words found
            break
            
        # Remove best_candidate
        c_idx = best_candidate
        active_code.remove(c_idx)
        removed += 1
        
        # Update cover_count
        for w_idx in coverage[c_idx]:
            cover_count[w_idx] -= 1
            
        # Optional: If we have time, we could try to add a new word to cover gaps?
        # No, we are removing. The initial code covers everything.
        # We are just pruning.
        
    # Construct final list
    final_code = [current_code[i] for i in active_code]
    
    # Final validation check (optional, for safety)
    # Ensure every word is covered
    # This is O(N * Space) which is 73 * 729 ~ 50k, very fast.
    if time.time() < deadline - 5:
        final_covered = np.zeros(len(all_words), dtype=bool)
        for c in final_code:
            for w_idx in coverage[current_code.index(c)]: # This is slow due to index lookup
                pass
        
        # Faster validation:
        # Rebuild coverage for final code
        final_covered_set = set()
        for c in final_code:
            # Find index in original current_code to reuse coverage set?
            # Or just recompute.
            c_list = list(c)
            # Distance 0
            final_covered_set.add(word_to_idx[c])
            # Distance 1
            for i in range(n):
                orig = c_list[i]
                for v in range(q):
                    if v == orig:
                        continue
                    new_c = c_list[:]
                    new_c[i] = v
                    nc = tuple(new_c)
                    if nc in word_to_idx:
                        final_covered_set.add(word_to_idx[nc])
                        
        if len(final_covered_set) < len(all_words):
            # Fallback to original base code if validation fails (should not happen)
            return base_code
        else:
            return [list(c) for c in final_code]
            
    return [list(c) for c in final_code]
