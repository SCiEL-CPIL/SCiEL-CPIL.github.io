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
    
    # Build a 2D array: neighbor_matrix[candidate][j] = neighbor index
    # Each candidate has exactly 1 + n*(q-1) neighbors for radius=1
    ball_size = 1 + n * (q - 1)
    neighbor_matrix = np.zeros((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
        neighbor_matrix[idx] = neighbors_np[idx]
    
    def greedy_cover(restart_seed=0, initial_codewords=None, symmetry_tie_break=False):
        """Greedy set cover with vectorized coverage counting."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        codewords = []
        # uncovered_mask[w] = True if word w is not yet covered
        uncovered_mask = np.ones(total_words, dtype=bool)
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    uncovered_mask[neighbor_matrix[cw]] = False
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while uncovered_mask.any() and time.time() < deadline:
            # Vectorized: for each candidate, count how many of its neighbors are uncovered
            # neighbor_matrix shape: (total_words, ball_size)
            # For each candidate row, sum over ball_size positions where neighbor is uncovered
            counts = uncovered_mask[neighbor_matrix].sum(axis=1)
            
            best_count = int(counts.max())
            if best_count == 0:
                break
            
            # Find all candidates with the best count
            best_candidates = np.where(counts == best_count)[0]
            
            # Tie-breaking: use random selection or symmetry
            if len(best_candidates) == 1:
                best_cw = int(best_candidates[0])
            elif symmetry_tie_break and len(best_candidates) > 1:
                # Prefer candidates whose neighbors overlap most with other best candidates
                # Quick heuristic: pick the one with the most neighbors in best_candidates set
                bc_set = set(int(x) for x in best_candidates)
                scores = np.array([len(set(neighbor_matrix[int(c)]) & bc_set) for c in best_candidates])
                best_cw = int(best_candidates[scores.argmax()])
            else:
                best_cw = int(rng.choice(best_candidates))
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_mask[neighbor_matrix[best_cw]] = False
        
        return codewords
    
    def remove_redundant(codewords):
        """Remove redundant codewords iteratively using vectorized coverage counts."""
        if time.time() > deadline:
            return codewords
        
        # Build coverage count: for each word, how many codewords cover it
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbor_matrix[cw]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                
                # Check if all words in ball(cw) are covered by at least 2 codewords
                # If so, cw is redundant
                if coverage[neighbor_matrix[cw]].min() >= 2:
                    # Remove cw and update coverage
                    coverage[neighbor_matrix[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search(codewords):
        """Try to swap codewords to improve coverage or reduce count."""
        if time.time() > deadline:
            return codewords
        
        # Try removing each codeword and see if we can re-cover with fewer
        # This is essentially the same as remove_redundant but more aggressive
        
        # Try: for each pair (i, j), see if removing i and adding some k reduces count
        # This is expensive. Instead, just do multiple passes of redundancy removal.
        
        return remove_redundant(codewords)
    
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
    reduced = remove_redundant(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds, some with symmetry tie-breaking
    for seed in range(60):
        if time.time() > deadline:
            break
        
        use_symmetry = (seed % 3 == 0)  # Every 3rd restart uses symmetry tie-breaking
        result = greedy_cover(restart_seed=seed, initial_codewords=None, symmetry_tie_break=use_symmetry)
        if result is not None:
            result = remove_redundant(result)
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
        use_symmetry = (seed % 3 == 0)
        result = greedy_cover(restart_seed=seed, initial_codewords=initial, symmetry_tie_break=use_symmetry)
        if result is not None:
            result = remove_redundant(result)
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
