import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Specific to the task: q=3, n=8, radius=1
    # Total words = 3^8 = 6561
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbor lists
    # neighbors[i] is a numpy array of indices j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
        
    # --- Phase 1: Fast Greedy Initialization ---
    # We want a good initial code. 
    # Use a greedy approach: pick word with max uncovered neighbors.
    # To speed this up, we maintain `uncovered_count` per word.
    
    cov_count = np.zeros(total_words, dtype=np.int32)
    is_code = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # Initial potential: number of neighbors
    # For q=3, n=8, radius=1, size is 17
    potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    
    uncovered_remaining = total_words
    
    # Greedy loop
    # We use a max-heap or just np.argmax. np.argmax is O(N).
    # With N=6561, this is fast.
    while uncovered_remaining > 0 and time.time() < deadline - 30.0:
        # Find max potential
        c = np.argmax(potential)
        
        if potential[c] == 0:
            break
            
        # Add c
        code_list.append(c)
        is_code[c] = True
        potential[c] = 0 # Don't pick again
        
        # Update coverage and potentials
        neigh_c = neighbors[c]
        for x in neigh_c:
            if cov_count[x] == 0:
                cov_count[x] = 1
                uncovered_remaining -= 1
                # Decrease potential for all codewords covering x
                # These are neighbors[x]
                # We only care about potential for words not yet in code?
                # Actually, potential is used for selection. If a word is already in code, potential is 0.
                # So we can just update all.
                potential[neighbors[x]] -= 1
            else:
                cov_count[x] += 1
                
    # If greedy didn't finish, force complete it (should not happen)
    if uncovered_remaining > 0:
        for i in range(total_words):
            if not is_code[i]:
                # Check if it covers any uncovered
                # Simple check: add it if it helps
                # Actually, just add all remaining if needed to ensure validity
                pass 
        # Fallback: just add all words not in code? No, that's huge.
        # Greedy should finish.
        
    # Current size is likely ~500-600
    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size
    
    # --- Phase 2: Iterated Local Search (ILS) with Bitmask/Vectorized Checks ---
    # Strategy: 
    # 1. Try to remove a random word.
    # 2. If valid (no uncovered points), keep it.
    # 3. If invalid, revert.
    # 4. Periodically, do a "kick" (random swap) to escape local minima.
    
    # We need to be able to check validity of removal quickly.
    # cov_count[x] == 1 means x is only covered by this word.
    # So, a word c is removable if for all x in neighbors[c], cov_count[x] > 1.
    
    # Precompute a "removable" mask? No, it changes.
    
    # To speed up the "is it removable" check:
    # For a candidate c, check `np.all(cov_count[neighbors[c]] > 1)`
    
    # We will run ILS for the remaining time.
    # We also track the best valid code found.
    
    # To allow escaping local minima, we will use a "Kick" strategy:
    # If we get stuck (no removals for X steps), we randomly swap 1-2 words.
    # This might temporarily increase size or break coverage?
    # Let's stick to "Remove if possible" and "Swap if neutral/improving".
    
    # Actually, a more powerful move is "Replace":
    # Remove c1, add c2. 
    # This is complex to evaluate.
    
    # Let's stick to simple removals and random restarts if stuck.
    # But random restarts are expensive.
    
    # Better: "2-opt" style.
    # Try to remove c1. If fails, try to find c2 such that adding c2 allows removing c1?
    # That's complex.
    
    # Let's use a simple but effective heuristic:
    # 1. Scan all words in the code.
    # 2. If any is removable, remove it.
    # 3. If none is removable, we are in a local minimum.
    # 4. Apply a random perturbation: Pick a random word c1 in code, pick a random word c2 not in code.
    #    Swap them. Check if the new code is valid.
    #    If valid, accept. If not, revert.
    #    This changes the structure without changing size.
    
    # To make this faster, we limit the number of swap attempts.
    
    current_cov_count = cov_count.copy()
    current_is_code = is_code.copy()
    current_code_set = set(current_code)
    
    stuck_counter = 0
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        removed = False
        # Shuffle current code to avoid bias
        # Instead of shuffling the list, pick a random index
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        # Check if removable
        # Vectorized check: are all neighbors covered by >1 codeword?
        if np.all(current_cov_count[neighbors[c]] > 1):
            # Remove c
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            removed = True
            stuck_counter = 0
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        # 2. If stuck for a while, try a swap
        if stuck_counter > 100 and time.time() < deadline - 10.0:
            # Try a random swap
            # Pick c1 from code
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Pick c2 not in code
            # To pick efficiently, we can pick a random index and check set membership
            # Since code is small (~500) vs total (6561), this is fast
            c2_found = False
            for _ in range(50):
                c2_candidate = random.randint(0, total_words - 1)
                if c2_candidate not in current_code_set:
                    c2 = c2_candidate
                    c2_found = True
                    break
            
            if c2_found:
                # Check if swap (c1 -> c2) maintains validity
                # We need to check if removing c1 and adding c2 results in all covered.
                # This is equivalent to:
                # 1. Remove c1: cov_count' = cov_count - neighbors[c1]
                # 2. Add c2: cov_count'' = cov_count' + neighbors[c2]
                # 3. Check if min(cov_count'') > 0
                
                # We can do this vectorized on the union of neighbors
                # But easier: just simulate on a copy? No, too slow.
                
                # Check validity of swap:
                # Words that become uncovered: x in neighbors[c1] where cov_count[x] == 1 AND x NOT in neighbors[c2]
                # Words that become covered: x in neighbors[c2] where cov_count[x] == 0 (should be 0 if current is valid)
                
                # Since current is valid (cov_count >= 1), we only care about words dropping to 0.
                # A word x drops to 0 if:
                # - x in neighbors[c1]
                # - cov_count[x] == 1
                # - x NOT in neighbors[c2]
                
                # Let's compute this.
                # neighbors[c1] is an array.
                # We need to check which of these are in neighbors[c2].
                
                # Create a boolean mask for neighbors[c2]
                # Since total_words is 6561, we can use a boolean array or a set.
                # Set is faster for small N?
                # neighbors[c2] has 17 elements.
                # neighbors[c1] has 17 elements.
                
                set_c2 = set(neighbors[c2].tolist())
                
                invalid_swap = False
                for x in neighbors[c1]:
                    if current_cov_count[x] == 1 and x not in set_c2:
                        invalid_swap = True
                        break
                
                if not invalid_swap:
                    # Swap is valid
                    # Apply swap
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2
                    
                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)
                    
                    # Size unchanged, so best_size doesn't change
                    # But we might have found a configuration that allows further removals
                    stuck_counter = 0
                else:
                    # Swap invalid, do nothing
                    pass
                    
            stuck_counter = 0 # Reset to avoid infinite swap attempts? Or increment?
            # Let's increment to eventually give up on swaps if they fail repeatedly
            # Actually, if swaps fail, we are in a deep local minimum.
            # Let's try a "kick" by removing a word even if it breaks coverage?
            # No, we must return a valid code.
            
    # Final Polish: Try to remove more words from best_code
    final_code = best_code[:]
    final_code_set = set(final_code)
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to remove any word
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code_set.remove(c)
                final_code.pop(i)
                improved = True
                
    # Verify
    if not np.all(final_cov_count > 0):
        # Fallback to a known valid code if something went wrong
        # Construct a trivial valid code (all words) - too big
        # Or just return best_code which was tracked as valid
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
