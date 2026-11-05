import numpy as np
import time
import itertools

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 5s margin before 120s kill
    
    if n <= 8 and radius == 1:
        total_words = q ** n
        sphere_size = 1 + n * (q - 1)
        
        # Precompute all neighbor indices for each word
        neighbors = np.full((total_words, sphere_size), -1, dtype=np.int32)
        for i in range(total_words):
            nbrs = [i]
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                current_val = (i // pow_q) % q
                for new_val in range(q):
                    if new_val != current_val:
                        nbrs.append(i + (new_val - current_val) * pow_q)
            neighbors[i] = nbrs

        # Greedy construction
        covered = np.zeros(total_words, dtype=bool)
        code_indices = []
        
        while time.time() < deadline - 30.0:
            gains = np.array([np.sum(~covered[neighbors[i]]) for i in range(total_words)])
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            code_indices.append(best_idx)
            covered[neighbors[best_idx]] = True
            
        # Local Search
        code_set = set(code_indices)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for ci in code_indices:
            cover_count[neighbors[ci]] += 1
            
        def is_redundant(c):
            return np.all(cover_count[neighbors[c]] >= 2)
            
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Phase 1: Remove redundant codewords
            for i in range(len(code_indices) - 1, -1, -1):
                c = code_indices[i]
                if is_redundant(c):
                    cover_count[neighbors[c]] -= 1
                    code_indices.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Phase 2: Swap to enable removal
            # Precompute critical words for each codeword in code
            # crit[c] = list of w in neighbors[c] such that cover_count[w] == 1
            
            # We only try swaps if we are stuck.
            # To speed up, we limit the search.
            
            # Find all non-code words
            # non_code = [w for w in range(total_words) if w not in code_set]
            
            # For each c_out in code:
            for c_out in code_indices:
                if time.time() > deadline - 15.0:
                    break
                
                # Get critical words for c_out
                nbrs_out = neighbors[c_out]
                crit_out = nbrs_out[cover_count[nbrs_out] == 1]
                
                if len(crit_out) == 0:
                    # This shouldn't happen if Phase 1 didn't remove it,
                    # but just in case, skip.
                    continue
                
                # Find c_in candidates that cover at least one word in crit_out
                # We can iterate all c_in not in code, but filter quickly.
                # To make it faster, we can precompute a mapping:
                # word -> list of codewords covering it?
                # That's expensive to maintain.
                
                # Let's just iterate all c_in not in code.
                # But we can break early if we find a good swap.
                
                for c_in in range(total_words):
                    if c_in in code_set:
                        continue
                    
                    # Check if c_in covers any word in crit_out
                    # If not, the swap doesn't help resolve the "critical" issue of c_out directly.
                    # However, it might help another codeword.
                    # But usually, we want to cover the "holes" left by c_out.
                    
                    # Quick check: does c_in share any neighbor with crit_out?
                    # nbrs_in = neighbors[c_in]
                    # if not np.any(np.isin(crit_out, nbrs_in)):
                    #     continue
                    
                    # Simulate swap
                    cover_count[neighbors[c_out]] -= 1
                    cover_count[neighbors[c_in]] += 1
                    
                    # Check if any codeword is now redundant
                    found_redundant = False
                    for c_check in code_indices:
                        if is_redundant(c_check):
                            found_redundant = True
                            break
                    
                    if found_redundant:
                        code_set.remove(c_out)
                        code_set.add(c_in)
                        idx_out = code_indices.index(c_out)
                        code_indices[idx_out] = c_in
                        improved = True
                        break # Break c_in loop
                    else:
                        cover_count[neighbors[c_out]] += 1
                        cover_count[neighbors[c_in]] -= 1
                
                if improved:
                    break # Break c_out loop
        
        result_words = []
        for idx in code_indices:
            w = []
            temp = idx
            for _ in range(n):
                w.append(temp % q)
                temp //= q
            w.reverse()
            result_words.append(w)
        
        return result_words
    else:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            base_n = 4
            base_code = tetracode
        else:
            base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            base_n = 4
        
        if n < base_n:
            return [list(range(n))]
        
        tails = list(itertools.product(range(q), repeat=n - base_n))
        return [h + list(tail) for h in base_code for tail in tails]
