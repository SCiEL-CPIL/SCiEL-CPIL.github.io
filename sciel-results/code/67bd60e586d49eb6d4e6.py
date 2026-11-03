import time
import sys

def construct(q, n, radius):
    # Parameters specific to the task instance
    # q=3, n=7, radius=1
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    # Precompute the "ball" (neighbors within distance radius) for each word
    # Represent each ball as a bitmask
    masks = [0] * total_words
    
    # Helper to convert index to tuple of digits
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return t

    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Generate masks
    # For each word, find all words within Hamming distance 1
    # Since n=7, q=3, distance 1 means changing 1 digit to 2 other values.
    # Total neighbors = 1 + n * (q-1) = 1 + 7*2 = 15.
    
    for i in range(total_words):
        t = idx_to_tuple(i)
        mask = 1 << i
        # Distance 0 is self
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t[pos] = new_val
                j = tuple_to_idx(t)
                mask |= (1 << j)
            t[pos] = orig_val
        masks[i] = mask

    # Greedy Set Cover
    covered = 0
    code = []
    # To speed up greedy, we can just iterate all candidates each time.
    # total_words is 2187.
    
    while covered != all_mask:
        best_gain = -1
        best_c = -1
        for c in range(total_words):
            # If c is already in code, gain is 0 (or we can skip, but checking is cheap)
            # Actually, if c is in code, masks[c] is subset of covered, so gain is 0.
            new_cover = covered | masks[c]
            # Calculate gain: popcount(new_cover) - popcount(covered)
            # In Python, int.bit_count() is available in 3.10+
            # Assuming Python 3.10+ is available (standard in modern environments)
            gain = (new_cover ^ covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1 or best_gain == 0:
            # Should not happen if covered != all_mask
            break
            
        code.append(best_c)
        covered |= masks[best_c]
        
    # Local Search to reduce code size
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # 1. Try to remove single redundant codewords
        for i in range(len(code)):
            if time.time() > deadline:
                break
            
            # Compute cover without code[i]
            temp_cover = 0
            for j in range(len(code)):
                if i != j:
                    temp_cover |= masks[code[j]]
            
            if temp_cover == all_mask:
                # code[i] is redundant
                code.pop(i)
                covered = temp_cover
                improved = True
                break  # Restart the loop as code changed
        
        if improved:
            continue
            
        # 2. Try 2-1 swaps (remove 2, add 1)
        # This is more expensive.
        # Iterate through pairs
        found_swap = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(code)):
                if time.time() > deadline:
                    break
                
                # Compute cover without code[i] and code[j]
                temp_cover = 0
                for k in range(len(code)):
                    if k != i and k != j:
                        temp_cover |= masks[code[k]]
                
                holes = all_mask & ~temp_cover
                if holes == 0:
                    # Both were redundant? Should have been caught in step 1
                    continue
                
                # Find a new codeword c_new that covers all holes
                # We can iterate through all possible c_new
                # Optimization: We only care about c_new that intersects holes
                for c_new in range(total_words):
                    if (masks[c_new] & holes) == holes:
                        # Check if c_new is already in code (excluding i, j)
                        # If it is, we are just moving it? No, we are replacing i,j with c_new.
                        # If c_new is already in code (at index k), then we are removing i,j and keeping k.
                        # This is valid.
                        # However, if c_new is in code, we might have just removed coverage that was needed?
                        # No, temp_cover was computed excluding i and j.
                        # If c_new is in code (index k != i,j), then masks[c_new] is already in temp_cover.
                        # So holes would be 0.
                        # Thus, if holes != 0, c_new cannot be in the remaining code.
                        
                        # Perform swap
                        # Remove i and j (order matters for popping)
                        if i > j:
                            i, j = j, i
                        code.pop(i)
                        code.pop(j) # j is now j-1 if i < j? 
                        # Let's do it safely:
                        # Actually, easier: create new list
                        new_code = []
                        for k in range(len(code) + 2): # original len was len(code)+2
                             # This logic is messy with indices.
                             pass
                        
                        # Safer way:
                        # Reconstruct code list
                        # We know i and j were in the old code.
                        # Let's just mark them for removal.
                        
                        # Let's restart the pair loop logic with a cleaner implementation
                        # We'll just break out and rebuild
                        pass
                        
                # The inner loop for c_new is 2187 iterations.
                # The outer loop is ~200*200/2 = 20000 pairs.
                # Total 20000 * 2187 = 43M iterations.
                # This might be too slow in Python if we do it naively.
                
                # Optimization: Precompute for each set of holes? No.
                # Optimization: Only check c_new if it covers at least one hole.
                # We can precompute a list of c_new for each hole bit? No.
                
                # Given the time limit, maybe we should restrict 2-1 search
                # or only do it if we have enough time.
                
                # Alternative: Randomized restarts?
                # Or just rely on single removals?
                # Single removals are very fast.
                # Greedy + Single Removals might get to ~190-200.
                # 243 -> 200 is a good score.
                
                # Let's try to optimize the 2-1 check.
                # If we find a swap, we accept it.
                
                # To speed up 2-1:
                # Instead of checking all c_new, we can check if there exists ANY c_new.
                # We can iterate c_new.
                
                # Let's implement the swap carefully.
                # If we find c_new:
                #   new_code = [c for k,c in enumerate(code) if k!=i and k!=j] + [c_new]
                #   code = new_code
                #   covered = temp_cover | masks[c_new]
                #   found_swap = True
                #   improved = True
                #   break loops
                
                # To make this faster, we can skip pairs if the holes are too large?
                # No, just run it.
                
                # Let's restructure the 2-1 loop to be efficient.
                pass

        # If no single removal and no 2-1 swap found, we are stuck.
        # We could try perturbation, but let's stick to this for now.
        
    return [idx_to_tuple(c) for c in code]

# Wait, the return type must be a list of codewords, where each codeword is a list of n integers.
# The function signature is construct(q, n, radius) -> list of codewords.

# Let's refine the 2-1 implementation to be correct and reasonably fast.

def construct(q, n, radius):
    total_words = q ** n
    all_mask = (1 << total_words) - 1
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return t

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    masks = [0] * total_words
    for i in range(total_words):
        t = idx_to_tuple(i)
        mask = 1 << i
        for pos in range(n):
            orig_val = t[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t[pos] = new_val
                j = tuple_to_idx(t)
                mask |= (1 << j)
            t[pos] = orig_val
        masks[i] = mask

    # Greedy
    covered = 0
    code = []
    while covered != all_mask:
        best_gain = -1
        best_c = -1
        for c in range(total_words):
            gain = ((covered | masks[c]) ^ covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain == 0:
            break
        code.append(best_c)
        covered |= masks[best_c]
        
    start_time = time.time()
    deadline = start_time + 140
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # 1. Single Removal
        for i in range(len(code)):
            if time.time() > deadline:
                break
            temp_cover = 0
            for j in range(len(code)):
                if i != j:
                    temp_cover |= masks[code[j]]
            if temp_cover == all_mask:
                code.pop(i)
                covered = temp_cover
                improved = True
                break
        
        if improved:
            continue
            
        # 2. 2-1 Swap
        # To make this feasible, we limit the number of pair checks per iteration
        # or use a heuristic.
        # Actually, 20000 pairs * 2187 checks is ~43M ops.
        # In Python, 43M simple ops might take ~5-10 seconds.
        # So one full pass of 2-1 search takes ~10s.
        # We have 140s. So we can do ~10 passes.
        # Each successful swap reduces size by 1.
        # If we start at 200, we can do 100 swaps.
        # So we might need 100 passes. 100 * 10s = 1000s. Too slow.
        
        # Optimization for 2-1:
        # We don't need to check all c_new for every pair.
        # We can precompute for each word w, which codewords cover it?
        # No, we need a single codeword covering a SET of holes.
        
        # Alternative: Randomized Local Search
        # Instead of exhaustive 2-1, try random pairs and random c_new?
        # Or try to remove a random codeword, and if it fails, try to repair by adding a random c_new?
        
        # Let's try a simpler 2-1:
        # For each i, remove code[i].
        # If holes exist, try to find a c_new that covers holes.
        # If found, we have a 1-1 swap (size same).
        # This doesn't reduce size.
        
        # To reduce size, we MUST remove 2 and add 1.
        
        # Maybe we can just do single removals and accept the result?
        # Greedy usually ends up around 200-220 for this problem.
        # 243 -> 210 is a score of 1.15.
        # 243 -> 186 is a score of 1.30.
        
        # Let's try to optimize the 2-1 search by only checking pairs where the union of their unique contributions is small?
        # Or just limit the time per 2-1 pass.
        
        # Let's do a time-boxed 2-1 search.
        pass_2_1_start = time.time()
        while time.time() - pass_2_1_start < 10: # 10s per pass
            if time.time() > deadline:
                break
            # Pick a random pair? Or sequential?
            # Sequential is better for determinism but slow.
            # Let's just do sequential but break if time runs out.
            
            found_swap = False
            for i in range(len(code)):
                if time.time() > deadline or time.time() - pass_2_1_start > 10:
                    break
                for j in range(i + 1, len(code)):
                    if time.time() > deadline or time.time() - pass_2_1_start > 10:
                        break
                    
                    temp_cover = 0
                    for k in range(len(code)):
                        if k != i and k != j:
                            temp_cover |= masks[code[k]]
                    
                    holes = all_mask & ~temp_cover
                    if holes == 0:
                        continue
                    
                    # Check for c_new
                    # Optimization: If holes is large, unlikely to be covered by one ball.
                    # Max ball size is 15.
                    # If popcount(holes) > 15, skip.
                    if holes.bit_count() > 15:
                        continue
                        
                    for c_new in range(total_words):
                        if (masks[c_new] & holes) == holes:
                            # Swap found
                            new_code = []
                            for k in range(len(code)):
                                if k != i and k != j:
                                    new_code.append(code[k])
                            new_code.append(c_new)
                            code = new_code
                            covered = temp_cover | masks[c_new]
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break
            
            if found_swap:
                break # Restart outer loop
            
    return [idx_to_tuple(c) for c in code]
