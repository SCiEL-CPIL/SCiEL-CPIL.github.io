"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # For the specific case q=3, n=6, radius=1
    total_words = q ** n
    
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute neighbor matrix once (shared across all greedy calls)
    neighbor_matrix = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_matrix[i, neighbors_np[i]] = True
    
    def greedy_cover(restart_seed=0, initial_codewords=None):
        """Greedy set cover with vectorized coverage counting."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        codewords = []
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
        
        # Uncovered words as a boolean array (True = uncovered)
        uncovered_arr = np.ones(total_words, dtype=bool)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = False
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while time.time() < deadline:
            # Check if all covered
            if not uncovered_arr.any():
                break
            
            # Count uncovered neighbors for each candidate
            counts = neighbor_matrix.dot(uncovered_arr.astype(np.int32))
            
            # Find the candidate with max count
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Among candidates with max count, pick one with some randomization
            # to diversify search
            max_mask = (counts == best_count)
            candidates = np.where(max_mask)[0]
            
            if len(candidates) == 1:
                best_cw = int(candidates[0])
            else:
                # Random tie-breaking for diversity
                best_cw = int(rng.choice(candidates.tolist()))
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = False
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively using coverage counts."""
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Check if all words in ball(cw) are covered by at least 2 codewords
                # If so, cw is redundant
                if coverage[neighbors_np[cw]].min() >= 2:
                    # Remove cw and update coverage
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def aggressive_reduction(codewords):
        """More aggressive reduction: try removing codewords and re-covering gaps."""
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        # First do standard redundancy removal
        codewords = remove_redundant(codewords)
        
        # Try to find codewords whose removal can be compensated by adding one other codeword
        # This is a 1-for-1 swap that reduces total count
        for _ in range(3):  # Try a few passes
            if time.time() > deadline:
                break
            
            found_swap = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                # Try removing codeword i
                cw = codewords[i]
                removed = codewords[:i] + codewords[i+1:]
                
                # Compute coverage without cw
                coverage = np.zeros(total_words, dtype=np.int32)
                for c in removed:
                    coverage[neighbors_np[c]] += 1
                
                # Find words not covered by remaining codewords
                uncovered = coverage == 0
                
                if not uncovered.any():
                    # cw was redundant, remove it
                    codewords = removed
                    found_swap = True
                    break
                
                # Find best replacement that covers the most uncovered words
                uncovered_arr = uncovered.astype(np.int32)
                counts = neighbor_matrix.dot(uncovered_arr)
                
                # Only consider candidates not already in the code
                for c in removed:
                    counts[c] = -1
                
                best_count = counts.max()
                
                if best_count >= uncovered.sum():
                    # This replacement covers all uncovered words
                    best_cw = int(counts.argmax())
                    codewords = removed + [best_cw]
                    found_swap = True
                    break
        
            if not found_swap:
                break
        
        return codewords
    
    def local_search(codewords):
        """Try to swap codewords to improve coverage or reduce count."""
        if time.time() > deadline:
            return codewords
        
        # Use aggressive reduction
        codewords = aggressive_reduction(codewords)
        
        return codewords
    
    # Strategy:
    # 1. Start with the tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords
    # 3. Try multiple random restarts of greedy to find better solutions
    # 4. Combine best results
    
    best_result = None
    best_count = float('inf')
    
    # First, try the tetracode construction with redundancy removal
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = aggressive_reduction(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    for seed in range(80):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None)
        if result is not None:
            result = aggressive_reduction(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(30):
        if time.time() > deadline:
            break
        
        # Random subset of tetracode as initial
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial)
        if result is not None:
            result = aggressive_reduction(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Final local search on best result
    if best_result is not None:
        best_result = local_search(best_result)
    
    # Convert back to lists
    if best_result is None:
        # Fallback: return the tetracode construction
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
