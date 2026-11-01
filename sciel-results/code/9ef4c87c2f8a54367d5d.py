import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases
    def fallback():
        if n == 0:
            return [[]]
        # Simple greedy for general case, though not optimized
        total = q ** n
        words = [list(c) for c in itertools.product(range(q), repeat=n)]
        # Map word to index
        def w2i(w):
            idx = 0
            for d in w:
                idx = idx * q + d
            return idx
        # Precompute neighbors
        nbs = []
        for w in words:
            nbs_set = {w2i(w)}
            for pos in range(n):
                orig = w[pos]
                for new in range(q):
                    if new == orig: continue
                    w2 = w[:]
                    w2[pos] = new
                    nbs_set.add(w2i(w2))
            nbs.append(nbs_set)
        
        uncovered = set(range(total))
        code = []
        while uncovered:
            best_c = -1
            max_cov = -1
            for i in range(total):
                cov = len(uncovered.intersection(nbs[i]))
                if cov > max_cov:
                    max_cov = cov
                    best_c = i
            if best_c == -1: break
            code.append(best_c)
            uncovered.difference_update(nbs[best_c])
        return [words[i] for i in code]

    if q != 3 or n != 7 or radius != 1:
        return fallback()

    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit

    # Precompute mapping and neighbors
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbor indices and bitmask
    neighbors_idx = []
    cover_masks = []
    
    for i in range(total_words):
        w = all_words[i]
        nbs = {i}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nb_idx = word_to_idx(w_copy)
                nbs.add(nb_idx)
        neighbors_idx.append(list(nbs))
        mask = 0
        for nb in nbs:
            mask |= (1 << nb)
        cover_masks.append(mask)
        
    # Helper to check if a code covers everything
    def is_valid(code_set):
        covered = 0
        for c in code_set:
            covered |= cover_masks[c]
        return covered == (1 << total_words) - 1

    def greedy_cover(randomize=False):
        uncovered = (1 << total_words) - 1
        code = []
        remaining = list(range(total_words))
        random.shuffle(remaining) # Shuffle for diversity even if not fully random choice
        
        while uncovered:
            best_idx = -1
            max_cov = -1
            candidates = []
            
            # Optimization: If uncovered count is small, we can be smarter, 
            # but for 2187 words, linear scan is fast enough in Python if bit ops are used.
            for c in remaining:
                cov = (uncovered & cover_masks[c]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    candidates = [c]
                elif cov == max_cov:
                    candidates.append(c)
                    
            if not candidates:
                break
                
            if randomize:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            remaining.remove(best_idx)
            
        return code

    def local_search(code_list):
        code_set = set(code_list)
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            cover_count[neighbors_idx[c]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Removals
            # Check if any codeword is redundant
            for c in list(code_set):
                # If all neighbors are covered by >1 codeword, we can remove c
                if cover_count[neighbors_idx[c]].min() > 1:
                    cover_count[neighbors_idx[c]] -= 1
                    code_set.remove(c)
                    improved = True
            
            if improved:
                continue
                
            # Pass 2: Swaps
            # Try to swap a codeword c with a non-codeword x
            # Condition: x covers all words that are ONLY covered by c (critical words)
            # And ideally x covers some new words or allows removal of another codeword
            # Simplified heuristic: Try random swaps that don't break coverage.
            
            # To speed up, we only try a limited number of swaps per iteration
            # And we focus on "critical" codewords if any exist, or random ones
            
            attempts = 50
            for _ in range(attempts):
                if not code_set:
                    break
                
                # Pick a codeword to replace
                # Bias towards codewords with few "unique" contributions? 
                # Or just random.
                c = random.choice(list(code_set))
                
                # Find words covered ONLY by c
                crit_words = [w for w in neighbors_idx[c] if cover_count[w] == 1]
                
                # If no critical words, c is redundant? 
                # Wait, if cover_count[w] == 1 for some w, c is critical for w.
                # If cover_count[w] > 1 for all w in N(c), c is redundant and should have been removed in Pass 1.
                # So if we reach here, crit_words should be non-empty.
                
                if not crit_words:
                    # Redundant, remove it (safety net)
                    cover_count[neighbors_idx[c]] -= 1
                    code_set.remove(c)
                    improved = True
                    break
                    
                # Try to find a replacement x
                # x must cover all crit_words
                # x should not be in code_set
                # x should ideally cover many words currently covered by 1 (to make them 2, allowing removal later?)
                # Or just any x that covers crit_words.
                
                # Sampling x:
                # Instead of checking all 2187, we can check a random subset
                # Or check neighbors of crit_words?
                # Let's just pick random x and check.
                
                found_swap = False
                for _ in range(20): # Limit inner loop
                    x = random.randint(0, total_words - 1)
                    if x in code_set:
                        continue
                    
                    # Check if x covers all crit_words
                    valid = True
                    # Optimize: check mask intersection
                    # mask_x = cover_masks[x]
                    # Need: mask_x & (1 << w) != 0 for all w in crit_words
                    # Equivalent: mask_x contains all bits in crit_mask
                    crit_mask = 0
                    for w in crit_words:
                        crit_mask |= (1 << w)
                        
                    if (cover_masks[x] & crit_mask) == crit_mask:
                        # Valid swap candidate
                        # Perform swap
                        cover_count[neighbors_idx[c]] -= 1
                        cover_count[neighbors_idx[x]] += 1
                        code_set.remove(c)
                        code_set.add(x)
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break
                    
        return list(code_set)

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    # 1. Deterministic Greedy
    cand = greedy_cover(randomize=False)
    cand = local_search(cand)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand
        
    # 2. Randomized Greedy + Local Search
    while time.time() < deadline:
        cand = greedy_cover(randomize=True)
        cand = local_search(cand)
        
        if len(cand) < best_size:
            best_size = len(cand)
            best_code = cand
            
        # Optional: Perturbation if stuck
        # If we have time, try a random perturbation on the best code
        if best_code and time.time() < deadline - 10:
            perturbed = list(best_code)
            if len(perturbed) > 2:
                # Remove a random codeword
                idx_to_remove = random.randint(0, len(perturbed) - 1)
                removed_cw = perturbed.pop(idx_to_remove)
                
                # Repair: Greedily add codewords to cover uncovered parts
                # Calculate current coverage
                covered_mask = 0
                for c in perturbed:
                    covered_mask |= cover_masks[c]
                    
                uncovered_mask = ((1 << total_words) - 1) & ~covered_mask
                
                # Greedily fill
                remaining_all = list(range(total_words))
                random.shuffle(remaining_all)
                while uncovered_mask:
                    best_c = -1
                    max_cov = -1
                    for c in remaining_all:
                        if c in perturbed: continue
                        cov = (uncovered_mask & cover_masks[c]).bit_count()
                        if cov > max_cov:
                            max_cov = cov
                            best_c = c
                    if best_c == -1: break
                    perturbed.append(best_c)
                    uncovered_mask &= ~cover_masks[best_c]
                    
                # Local search on perturbed
                perturbed = local_search(perturbed)
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed

    if best_code is None:
        # Fallback if something went wrong
        best_code = greedy_cover(randomize=False)
        best_code = local_search(best_code)
        
    return [all_words[i] for i in best_code]
