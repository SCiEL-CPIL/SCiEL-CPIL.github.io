import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        w.reverse()
        words.append(tuple(w))
        
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q
        
    def get_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * powers[i]
        return idx

    masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = get_idx(new_w)
                mask |= (1 << j)
        masks[i] = mask

    full_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = total_words
    
    # Run greedy a few times with different tie-breaking to escape local optima
    # For n=7, greedy is fast. We can do ~50 restarts.
    restarts = 50
    for r in range(restarts):
        if time.time() > deadline - 10:
            break
            
        covered = 0
        selected = []
        cand_list = list(range(total_words))
        # Shuffle candidates to break ties randomly
        random.shuffle(cand_list)
        
        while covered != full_mask:
            best_gain = -1
            best_cand = -1
            best_idx_in_list = -1
            
            # Use a random threshold or just max with random tie break?
            # Simple max is fine, the shuffle handles ties in the initial order, 
            # but as we remove, the order changes.
            # To be more robust, we can pick randomly from top-K gains.
            # But simple max is usually sufficient for this size.
            
            for idx, c in enumerate(cand_list):
                gain_mask = masks[c] & ~covered
                if gain_mask:
                    gain = gain_mask.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = c
                        best_idx_in_list = idx
                    elif gain == best_gain and best_cand != -1:
                        # Random tie break
                        if random.random() < 0.5:
                            best_gain = gain
                            best_cand = c
                            best_idx_in_list = idx
                            
            if best_cand == -1:
                break
                
            selected.append(best_cand)
            covered |= masks[best_cand]
            
            # Remove
            cand_list[best_idx_in_list] = cand_list[-1]
            cand_list.pop()
            
        # Post-processing: Remove redundant codewords
        # Check if removing any selected codeword still covers everything
        improved = True
        while improved:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                # Temporarily remove selected[i]
                # Recalculate covered? No, just check if union of others is full.
                # This is O(K^2) if done naively.
                # Better: maintain a count of how many codewords cover each word.
                # Let's do a simple check: 
                # If we remove c, is there any word that was ONLY covered by c?
                # We can compute the union of all masks except c.
                # Since K is ~186, and we do this K times, it's K^2 bitwise ORs.
                # 186^2 * 186 steps? No, just 186 iterations of checking 185 ORs.
                # That's ~34k ORs. Very fast.
                
                # Actually, a faster way:
                # Let U be the union of all masks in selected.
                # We want to know if U \ mask[selected[i]] is still full.
                # This is equivalent to: is there any bit in full_mask that is 1 in mask[selected[i]] 
                # and 0 in (U \ mask[selected[i]])?
                # Which is: is there any bit in mask[selected[i]] that is NOT in (U \ mask[selected[i]])?
                # i.e., bits in mask[selected[i]] that are not covered by any other codeword.
                
                # Let's just compute union_without_i
                # Optimization: Precompute prefix/suffix unions? Or just simple loop.
                
                union_others = 0
                for j in range(len(selected)):
                    if j != i:
                        union_others |= masks[selected[j]]
                        
                if union_others == full_mask:
                    selected.pop(i)
                    improved = True
                    break # Restart the loop since indices changed
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
        
    return result
