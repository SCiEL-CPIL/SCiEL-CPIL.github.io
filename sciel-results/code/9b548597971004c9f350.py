"""Improved covering code construction for K_q(n, radius) with local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin within 150s limit
    
    total_words = q ** n
    
    # Precompute all words
    all_words = list(product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor index lists for each word (words within Hamming distance <= radius)
    # For radius=1, this is the word itself plus all words at distance 1
    neighbor_indices = []
    for w in all_words:
        nbs = [w]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = list(w)
                    new_word[pos] = val
                    nbs.append(tuple(new_word))
        neighbor_indices.append([word_to_idx[nei] for nei in nbs])
    
    # Coverage count array
    coverage = np.zeros(total_words, dtype=np.int32)
    
    # --- Step 1: Start with tetracode product construction ---
    # Tetracode: 9 words of length 4, perfect code radius 1 over Z_3
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    initial_codewords = []
    if n >= 4 and q == 3:
        for h in tetracode:
            for tail in product(range(q), repeat=n - 4):
                initial_codewords.append(tuple(h + list(tail)))
    else:
        # Fallback: greedy construction
        uncovered = set(range(total_words))
        while uncovered and time.time() < deadline - 60.0:
            # Pick word covering most uncovered words
            best_idx = -1
            best_count = 0
            for idx in list(uncovered):
                count = sum(1 for ni in neighbor_indices[idx] if ni in uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            initial_codewords.append(all_words[best_idx])
            for ni in neighbor_indices[best_idx]:
                uncovered.discard(ni)
    
    # Convert to indices
    initial_indices = [word_to_idx[w] for w in initial_codewords if w in word_to_idx]
    
    # Build coverage from initial code
    for idx in initial_indices:
        for ni in neighbor_indices[idx]:
            coverage[ni] += 1
    
    codeword_indices = list(initial_indices)
    in_code = set(codeword_indices)
    
    def try_remove(i):
        """Try to remove codeword at position i. Returns True if removed."""
        if time.time() > deadline - 5.0:
            return False
        idx = codeword_indices[i]
        # Check if all neighbors of this codeword are covered by others
        for ni in neighbor_indices[idx]:
            if coverage[ni] <= 1:
                return False
        # Remove it
        for ni in neighbor_indices[idx]:
            coverage[ni] -= 1
        codeword_indices.pop(i)
        in_code.discard(idx)
        return True
    
    def try_swap(i, new_idx):
        """Try to swap codeword at position i with new_idx. Returns True if swapped."""
        if time.time() > deadline - 5.0:
            return False
        old_idx = codeword_indices[i]
        if new_idx == old_idx or new_idx in in_code:
            return False
        
        # Check: after removing old and adding new, is everything covered?
        # First, check which words would become uncovered if we remove old
        would_uncover = []
        for ni in neighbor_indices[old_idx]:
            if coverage[ni] <= 1:
                would_uncover.append(ni)
        
        # Check if new covers all those words
        new_neighbors = set(neighbor_indices[new_idx])
        for w in would_uncover:
            if w not in new_neighbors:
                return False
        
        # Perform swap
        for ni in neighbor_indices[old_idx]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx]:
            coverage[ni] += 1
        codeword_indices[i] = new_idx
        in_code.discard(old_idx)
        in_code.add(new_idx)
        return True
    
    def find_candidates_covering(words_to_cover, max_candidates=100):
        """Find codewords that cover all words in words_to_cover."""
        if not words_to_cover:
            return []
        
        # Start with candidates that cover the first word
        first_word = words_to_cover[0]
        candidate_set = set(neighbor_indices[first_word])
        
        # Intersect with candidates covering each subsequent word
        for w in words_to_cover[1:]:
            candidate_set = candidate_set & set(neighbor_indices[w])
            if len(candidate_set) == 0:
                return []
        
        # Filter out existing codewords
        candidates = [c for c in candidate_set if c not in in_code]
        return candidates[:max_candidates]
    
    # Iterative improvement: repeatedly try removals and swaps
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        
        # Phase 1: Try removals (in random order for diversity)
        order = list(range(len(codeword_indices)))
        random.shuffle(order)
        for i in sorted(order, key=lambda x: -x):  # Process in reverse to maintain indices
            if time.time() > deadline - 15.0:
                break
            if try_remove(i):
                improved = True
        
        if time.time() > deadline - 15.0:
            break
        
        # Phase 2: Try swaps - for each codeword, try replacing with targeted candidates
        for i in range(len(codeword_indices)):
            if time.time() > deadline - 15.0:
                break
            
            old_idx = codeword_indices[i]
            
            # Find words that are ONLY covered by this codeword
            exclusive_words = []
            for ni in neighbor_indices[old_idx]:
                if coverage[ni] == 1:
                    exclusive_words.append(ni)
            
            if not exclusive_words:
                continue
            
            # Find candidates that cover all exclusive words
            candidates = find_candidates_covering(exclusive_words, max_candidates=50)
            
            # Try each candidate
            for cand in candidates:
                if time.time() > deadline - 15.0:
                    break
                if try_swap(i, cand):
                    improved = True
                    break
            
            # If no targeted swap worked, try some random swaps as fallback
            if not improved and time.time() < deadline - 15.0:
                for _ in range(10):
                    if time.time() > deadline - 15.0:
                        break
                    cand = random.randint(0, total_words - 1)
                    if cand in in_code:
                        continue
                    if try_swap(i, cand):
                        improved = True
                        break
    
    # --- Step 3: Final cleanup pass ---
    if time.time() < deadline - 5.0:
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            try_remove(i)
    
    # Convert back to lists
    result = [list(all_words[idx]) for idx in codeword_indices]
    
    return result
