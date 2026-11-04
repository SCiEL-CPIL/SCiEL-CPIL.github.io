"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Problem parameters
    N = q ** n  # 2187 for q=3, n=7
    
    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word
    
    # Precompute all words
    words = [idx_to_word(i) for i in range(N)]
    
    # Build word to index mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Compute neighbors (ball of radius 1) as bitsets
    neighbors = [0] * N
    for i in range(N):
        w_i = words[i]
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        neighbors[i] = mask
    
    full_mask = (1 << N) - 1
    best_code = None
    best_size = float('inf')
    
    def greedy_cover(uncovered_bits, code, time_limit):
        """Greedy set cover to add codewords until all covered."""
        code = list(code)
        
        while uncovered_bits and time.time() < time_limit:
            best_candidate = -1
            max_new_cov = 0
            candidates = []
            
            # Iterate through all potential candidates
            # Optimization: We could skip candidates already in code if we want strict addition,
            # but re-adding is harmless if we just track the set. 
            # However, standard greedy adds new ones. 
            # Since N is small (2187), scanning all is fast with bitsets.
            
            for c in range(N):
                # new coverage = neighbors[c] & uncovered_bits
                new_cov = neighbors[c] & uncovered_bits
                if new_cov:
                    count = new_cov.bit_count()
                    if count > max_new_cov:
                        max_new_cov = count
                        best_candidate = c
                        candidates = [c]
                    elif count == max_new_cov:
                        candidates.append(c)
            
            if best_candidate == -1:
                break
            
            chosen = random.choice(candidates)
            code.append(chosen)
            uncovered_bits &= ~neighbors[chosen]
        
        return code
    
    def prune_redundant(code, time_limit):
        """Remove redundant codewords."""
        if not code:
            return code
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            to_remove = []
            
            # Precompute union of all neighbors for efficiency?
            # Or just compute 'others_cover' for each candidate?
            # K is ~180. K^2 bit ops is fine.
            
            # To optimize, we can compute total_cover first
            total_cover = 0
            for c in code:
                total_cover |= neighbors[c]
            
            for idx, c in enumerate(code):
                # Check if neighbors[c] is subset of (total_cover ^ neighbors[c])?
                # No, total_cover includes neighbors[c].
                # We need to know if there is any point in neighbors[c] that is ONLY covered by c.
                # i.e., is there a point p in neighbors[c] such that no other c' in code covers p?
                # This is equivalent to: neighbors[c] & ~(OR of neighbors[k] for k != c) == 0
                
                # Efficient way:
                # others_cover = total_cover
                # We can't just XOR because of overlaps.
                # But we can iterate? No, that's slow.
                
                # Alternative:
                # If we maintain counts, it's O(N) per check.
                # With bitsets, checking if a specific point is covered by others is hard without iterating points.
                
                # Let's stick to a slightly slower but robust bitset check:
                # For each c, compute union of neighbors of all other c's.
                # This is O(K) big-int ORs per c. Total O(K^2).
                # 180^2 = 32400 operations. Each op is on ~35 64-bit words. Very fast.
                
                others_cover = 0
                for k, c_k in enumerate(code):
                    if k != idx:
                        others_cover |= neighbors[c_k]
                
                # If neighbors[c] is subset of others_cover, it's redundant
                if (neighbors[c] & ~others_cover) == 0:
                    to_remove.append(idx)
            
            if to_remove:
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True
        
        return code
    
    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code
        
        code = list(code)
        # Remove random codewords
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)
        
        # Find uncovered bits
        covered_bits = 0
        for c in code:
            covered_bits |= neighbors[c]
        
        uncovered_bits = full_mask & ~covered_bits
        
        # Repair
        code = greedy_cover(uncovered_bits, code, time_limit)
        code = prune_redundant(code, time_limit)
        
        return code
    
    # Run multiple iterations with different strategies
    for iteration in range(100):  # Increased iterations due to speedup
        if time.time() > deadline - 5:
            break
        
        # Start with greedy
        code = greedy_cover(full_mask, [], deadline - 2)
        code = prune_redundant(code, deadline - 2)
        
        # Try kicks
        # We can try more aggressive kicks now
        for num_kicks in [1, 2, 3, 4, 5, 6, 8, 10]:
            if time.time() > deadline - 2:
                break
            code = kick_and_repair(code, num_kicks, deadline - 2)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            if best_size <= 186:
                break
    
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(words[c]) for c in best_code]
# EVOLVE-BLOCK-END
