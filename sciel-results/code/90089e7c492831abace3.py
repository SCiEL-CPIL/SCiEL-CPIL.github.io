import itertools
import random
import time

def construct(q, n, radius):
    # Task specific parameters
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task is specific
        # Simple greedy for general case
        pass 

    # Specific to 3, 7, 1
    total_words = q ** n
    # Map word tuple to index
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    # Precompute balls: ball[i] is a bitmask of all words within distance radius of idx i
    # Using Python integers as bitsets
    balls = [0] * total_words
    for i in range(total_words):
        center = idx_to_tuple(i)
        mask = 0
        # Generate all words within distance radius
        # For n=7, radius=1, it's center + 7*2 neighbors
        for dist in range(radius + 1):
            # This is a bit slow for precomputation if done via recursion for general n, 
            # but for n=7, radius=1, we can hardcode or use product
            if dist == 0:
                mask |= (1 << i)
            elif dist == 1:
                for pos in range(n):
                    for val in range(q):
                        if val == center[pos]:
                            continue
                        neighbor = list(center)
                        neighbor[pos] = val
                        n_idx = tuple_to_idx(tuple(neighbor))
                        mask |= (1 << n_idx)
        balls[i] = mask

    full_mask = (1 << total_words) - 1
    
    # Greedy Initialization
    code = []
    covered = 0
    # To speed up greedy, we can just iterate and pick best
    # But 2187 is small.
    # We want to pick c that maximizes (balls[c] & ~covered)
    
    # A faster greedy:
    # While covered != full_mask:
    #   best_c = -1
    #   best_gain = -1
    #   for c in range(total_words):
    #     gain = bin(balls[c] & ~covered).count('1')
    #     if gain > best_gain:
    #       best_gain = gain
    #       best_c = c
    #   code.append(best_c)
    #   covered |= balls[best_c]
    
    # This is O(3^n * 3^n) roughly if we check all c each time. 2187^2 ~ 4.7M. Fast enough.
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # 1. Greedy Construction
    covered = 0
    while covered != full_mask:
        best_c = -1
        best_gain = -1
        # Optimization: Only consider candidates that add at least 1 new word
        # We can precompute nothing, just scan
        for c in range(total_words):
            new_words = balls[c] & ~covered
            if new_words:
                # Count bits is fast in Python for ints
                gain = new_words.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
        if best_c == -1:
            break # Should not happen if full_mask is reachable
        code.append(best_c)
        covered |= balls[best_c]
        
    # Initial size is likely around 180-200.
    
    # Helper to check if code is a valid cover
    def is_valid(code_list):
        cov = 0
        for c in code_list:
            cov |= balls[c]
        return cov == full_mask

    # Helper to find redundant codewords
    def find_redundant(code_list):
        redundant = []
        N = len(code_list)
        # For each c, check if removing it leaves coverage intact
        # Optimization: Compute total coverage once? 
        # No, we need coverage of others.
        # N is small (~200). N^2 * bitops is fine.
        for i, c in enumerate(code_list):
            # Calculate coverage of all others
            cov_others = 0
            for j, c2 in enumerate(code_list):
                if i != j:
                    cov_others |= balls[c2]
            if (cov_others | balls[c]) == cov_others:
                redundant.append(i)
        return redundant

    # Main Local Search Loop
    best_code = list(code)
    best_size = len(code)
    
    while time.time() < deadline:
        # Phase 1: Remove Redundant
        changed = True
        while changed:
            changed = False
            red_idx = find_redundant(code)
            if red_idx:
                # Remove the first redundant one found (or all?)
                # Removing one at a time is safer for subsequent checks
                idx_to_remove = red_idx[0]
                code.pop(idx_to_remove)
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                    
        # Phase 2: 1-for-1 Swaps
        # Try to swap c with c_new
        # To avoid O(N * 3^n) being too slow if N is large, 
        # but N~200, 3^n=2187. 400k ops per pass. Very fast.
        
        # Shuffle to escape local minima
        c_indices = list(range(len(code)))
        random.shuffle(c_indices)
        
        for i in c_indices:
            if time.time() > deadline:
                break
            c_old = code[i]
            # Try all possible c_new
            # We can skip c_new that are already in code (unless we want to swap with identical, which is useless)
            # We should also prioritize c_new that cover more "critical" words
            # Simple: try all c_new
            for c_new in range(total_words):
                if c_new == c_old:
                    continue
                if c_new in code:
                    continue
                
                # Check validity of swap
                # Remove c_old, add c_new
                # cov_others = OR of balls of all c in code except c_old
                cov_others = 0
                for j, c in enumerate(code):
                    if j != i:
                        cov_others |= balls[c]
                
                new_cov = cov_others | balls[c_new]
                if new_cov == full_mask:
                    # Valid swap
                    code[i] = c_new
                    changed = True
                    break # Restart phase 1/2? Or just continue? 
                    # Restarting phase 1 is good to pick up new redundancies
                    
        if changed:
            # After a swap, check for redundancies again
            red_idx = find_redundant(code)
            if red_idx:
                idx_to_remove = red_idx[0]
                code.pop(idx_to_remove)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            
            # If we did a swap, we should probably restart the loop to re-evaluate
            # But to save time, we can just continue the inner loop or restart the while
            # Restarting the while is safer for convergence
            continue

        # Phase 3: 2-for-1 Swaps (if time permits)
        # Try to remove 2 and add 1
        # This is expensive: O(N^2 * 3^n)
        # N~200, N^2=40000, 3^n=2187. 87M ops. Might be too slow for a full pass in Python.
        # We can try random pairs.
        
        if time.time() < deadline - 5:
            # Try a few random 2-for-1 moves
            attempts = 0
            while attempts < 100 and time.time() < deadline - 5:
                attempts += 1
                if len(code) < 2:
                    break
                i, j = random.sample(range(len(code)), 2)
                c1 = code[i]
                c2 = code[j]
                
                # Calculate coverage of others
                cov_others = 0
                for k, c in enumerate(code):
                    if k != i and k != j:
                        cov_others |= balls[c]
                
                # What is missing?
                missing = full_mask & ~cov_others
                
                # If missing is 0, we can just remove both
                if missing == 0:
                    # Remove j and i (j > i usually, so remove j first to keep indices valid)
                    if i > j:
                        code.pop(i)
                        code.pop(j)
                    else:
                        code.pop(j)
                        code.pop(i)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break # Restart main loop
                    
                # Else, we need a c_new that covers 'missing'
                # Find c_new such that (balls[c_new] & missing) == missing
                # i.e., balls[c_new] is a superset of missing
                found = False
                # We can iterate c_new
                for c_new in range(total_words):
                    if c_new == c1 or c_new == c2:
                        continue
                    # Check if c_new covers all missing words
                    if (balls[c_new] & missing) == missing:
                        # Valid 2-for-1
                        # Remove c1, c2, add c_new
                        if i > j:
                            code.pop(i)
                            code.pop(j)
                        else:
                            code.pop(j)
                            code.pop(i)
                        code.append(c_new)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = list(code)
                        found = True
                        break
                
                if found:
                    break # Restart main loop
                    
    # Ensure we return a valid code
    # The best_code should be valid by construction, but let's verify
    # If best_code is not valid (shouldn't happen), fall back to seed
    if not is_valid(best_code):
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [tuple_to_idx(tuple(h + list(tail))) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to lists of ints
    result = [list(idx_to_tuple(idx)) for idx in best_code]
    return result

# EVOLVE-BLOCK-END
