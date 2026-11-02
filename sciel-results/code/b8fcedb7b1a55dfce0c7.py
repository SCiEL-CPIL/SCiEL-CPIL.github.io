import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Problem specific: q=3, n=8, radius=1
    # If parameters differ, fall back to seed logic or generic greedy
    if q != 3 or n != 8 or radius != 1:
        # Fallback: Direct sum of tetracode if possible, else simple greedy
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            return code
        else:
            # Generic greedy (slow, but safe)
            return _generic_greedy(q, n, radius)

    total_words = q ** n
    # Map word (list of ints) to integer index
    # index = w[0]*3^7 + w[1]*3^6 + ... + w[7]
    
    # Precompute neighbors for all words in the space
    # neighbors[i] is a list of indices
    neighbors = np.empty((total_words, 17), dtype=np.int32)
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    for i in range(total_words):
        # Convert i to base-q digits
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        # w[0] is least significant? No, powers list:
        # index = sum(w[j] * powers[j])
        # So w[j] corresponds to power[j]
        # Let's stick to a consistent mapping.
        # Let's use standard base conversion where w[0] is most significant.
        # index = w[0]*q^(n-1) + ... + w[n-1]
        
        # Re-do mapping for clarity
        # index -> word
        # word -> index
        
        # Let's generate all words as tuples and map them
        pass

    # Efficient mapping
    # Create a list of all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    # For each word, neighbors are itself + words with 1 position changed
    neighbors_list = []
    for w in all_words:
        w_tuple = w
        idx = word_to_idx[w_tuple]
        nbrs = [idx]
        for pos in range(n):
            for val in range(q):
                if w_tuple[pos] == val:
                    continue
                new_w_list = list(w_tuple)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                nbrs.append(word_to_idx[new_w])
        neighbors_list.append(nbrs)
        
    neighbors_arr = np.array(neighbors_list, dtype=np.int32)

    # Initial code: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code_indices.append(word_to_idx[w])
            
    code = set(initial_code_indices)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize cover_count
    for c in code:
        cover_count[neighbors_arr[c]] += 1
        
    # Local Search
    # State: code (set), cover_count (array)
    
    def try_remove(c):
        # Check if all neighbors of c are covered by > 1 codeword
        # i.e., cover_count[neighbors_arr[c]] > 1
        if np.all(cover_count[neighbors_arr[c]] > 1):
            # Remove c
            cover_count[neighbors_arr[c]] -= 1
            code.remove(c)
            return True
        return False

    def try_swap(c, c_new):
        # Check if swapping c with c_new is valid
        # Words covered by c but not c_new must be covered by others
        # Words covered by c_new but not c will be covered by c_new
        # We need: for all w in N(c) \ N(c_new), cover_count[w] > 1
        
        # Optimization: 
        # gain = N(c_new) \ N(c)
        # loss = N(c) \ N(c_new)
        # If loss is covered by others, swap is valid.
        
        # Simple check:
        # Create a mask? 
        # Since 17 neighbors, just loop.
        
        # Get neighbors of c and c_new
        nbrs_c = neighbors_arr[c]
        nbrs_c_new = neighbors_arr[c_new]
        
        # Check loss
        # For each w in nbrs_c, if w not in nbrs_c_new, check cover_count[w] > 1
        # To avoid set operations, we can use a temporary array or just check membership.
        # 17x17 = 289 checks. Fast enough.
        
        nbrs_c_set = set(nbrs_c)
        nbrs_c_new_set = set(nbrs_c_new)
        
        for w in nbrs_c:
            if w not in nbrs_c_new_set:
                if cover_count[w] <= 1:
                    return False
        return True

    # Main loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant words
        # Shuffle code to avoid bias?
        code_list = list(code)
        np.random.shuffle(code_list)
        
        for c in code_list:
            if c not in code: continue # might have been removed? No, we remove one by one.
            if try_remove(c):
                improved = True
                # Don't break, try to remove more
        
        # 2. If no removals, try swaps to enable removals
        if not improved and time.time() < deadline:
            code_list = list(code)
            np.random.shuffle(code_list)
            
            # Candidate new words: words not in code
            # We can't iterate all 6561 for each c (486*6561 is ~3M, ok for a few passes)
            # But let's be smarter. 
            # We want to find c_new that covers "loss" of c.
            
            for c in code_list:
                if c not in code: continue
                if time.time() > deadline: break
                
                nbrs_c = neighbors_arr[c]
                nbrs_c_set = set(nbrs_c)
                
                # Find words w in N(c) that are covered only by c (cover_count[w]==1)
                # These MUST be covered by the new word c_new.
                critical_words = [w for w in nbrs_c if cover_count[w] == 1]
                
                if not critical_words:
                    # If no critical words, we can remove c (should have been caught in step 1)
                    # But maybe step 1 missed it due to order? 
                    # If step 1 didn't remove it, it means some w in N(c) has cover_count==1.
                    # So critical_words should not be empty if try_remove failed.
                    pass
                
                # We need a c_new such that:
                # 1. c_new is not in code
                # 2. critical_words subset of N(c_new)
                # 3. For all w in N(c) \ N(c_new), cover_count[w] > 1
                
                # Strategy: Pick a critical word w_crit.
                # c_new must be in N(w_crit).
                # N(w_crit) has 17 words.
                # Try all c_new in N(w_crit) that are not in code.
                
                if critical_words:
                    w_crit = critical_words[0]
                    candidates = neighbors_arr[w_crit]
                    
                    for c_new in candidates:
                        if c_new in code:
                            continue
                        if c_new == c:
                            continue
                        
                        if try_swap(c, c_new):
                            # Perform swap
                            cover_count[neighbors_arr[c]] -= 1
                            cover_count[neighbors_arr[c_new]] += 1
                            code.remove(c)
                            code.add(c_new)
                            improved = True
                            break # Try next c
                    
                    if improved:
                        # Restart removal phase? Or continue?
                        # Continuing might be ok, but removal phase is faster.
                        break

    # Convert indices back to words
    final_code = [list(all_words[idx]) for idx in code]
    
    # Optional: Final check
    # return final_code
    return final_code

def _generic_greedy(q, n, radius):
    # Fallback for non-3,8,1 cases
    # Simple greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute distance? Too slow for large n.
    # Just use greedy: pick word covering most uncovered.
    
    covered = set()
    code = []
    remaining = set(range(len(all_words)))
    
    # This is O(N^2) or worse. Only for small n.
    if len(all_words) > 10000:
        # Return empty or seed-like
        return []
        
    while remaining:
        best_word = None
        best_count = -1
        for w in all_words:
            idx = word_to_idx[w]
            if idx in covered: # w is already covered? No, w is a candidate codeword.
                pass
            # Count how many uncovered words w covers
            # Neighbors of w within radius
            cnt = 0
            # Generate neighbors
            w_list = list(w)
            for delta_pos in range(n):
                for delta_val in range(q):
                    if w_list[delta_pos] == delta_val:
                        continue
                    new_w_list = w_list[:]
                    new_w_list[delta_pos] = delta_val
                    new_w = tuple(new_w_list)
                    if new_w in remaining:
                        cnt += 1
            # Also w itself
            if w in remaining:
                cnt += 1
            
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None:
            break
            
        code.append(list(best_word))
        # Update remaining
        best_word_list = list(best_word)
        # Remove best_word and its neighbors from remaining
        to_remove = set()
        to_remove.add(best_word)
        for delta_pos in range(n):
            for delta_val in range(q):
                if best_word_list[delta_pos] == delta_val:
                    continue
                new_w_list = best_word_list[:]
                new_w_list[delta_pos] = delta_val
                to_remove.add(tuple(new_w_list))
                
        remaining -= to_remove
        
    return code
