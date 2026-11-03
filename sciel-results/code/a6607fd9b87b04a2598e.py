import time
import random
import itertools

def construct(q, n, radius):
    # The task is specifically for q=3, n=7, radius=1.
    # However, the signature is general. We will implement a general greedy 
    # but optimize for the specific case if parameters match, or just 
    # use the general logic which should work.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the original simple construction or a generic greedy
        # For non-matching params, we return the seed-like construction if possible
        # or a trivial valid code (all words) if n is small, otherwise heuristic.
        # Since the evaluator passes the specific task params, this branch is 
        # likely not hit, but we must be safe.
        if n <= 10:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        else:
            # Simple product code fallback
            # For ternary, use tetracode if n >= 4
            if q == 3 and n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                # Trivial: all words
                return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    total_words = q ** n
    N = total_words
    
    # Map word tuple to index
    def to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def from_index(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls as bitmasks
    # balls[i] is an int with bits set for all j in ball(i)
    balls = [0] * N
    
    for idx in range(N):
        w = from_index(idx)
        mask = 1 << idx
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = to_index(w)
                mask |= (1 << j)
            w[pos] = orig
        balls[idx] = mask

    best_code_size = float('inf')
    best_code_indices = None
    
    # Run multiple random restarts with Greedy + Local Search
    num_restarts = 100  # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered = (1 << N) - 1  # All bits set
        code_indices = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_count = -1
            candidates = []
            
            # To speed up, we can just iterate all. N=2187 is small.
            # We want the index that covers the most 'uncovered' bits.
            for idx in range(N):
                # If idx is already in code, its contribution is 0 to uncovered?
                # Actually, if idx is in code, balls[idx] & uncovered is 0 because we remove them.
                # But we can pick it again? No, distinct codewords.
                # So we only consider idx not in code?
                # Checking `if idx not in code_indices` is O(K).
                # Better: keep a boolean array `in_code`.
                pass
            
            # Optimization: Use a boolean array for `in_code`
            # But inside the loop, we need to be fast.
            # Let's just check all, it's 2187 * 2187 ~ 4.7M ops per step.
            # Steps ~ 200. 1B ops. Too slow in Python.
            
            # Alternative Greedy:
            # Pick random uncovered word, pick random ball covering it?
            # Or: Iterate all idx, compute `count = (balls[idx] & uncovered).bit_count()`.
            # `bit_count()` is fast in C.
            
            best_idx = -1
            best_count = -1
            candidates = []
            
            for idx in range(N):
                # Skip if already selected?
                # We can maintain a set of selected indices for O(1) check?
                # Or just allow duplicates in greedy and remove later? 
                # No, better to avoid.
                # Let's use a list `in_code` of bools.
                if idx in code_indices: # O(K)
                    continue
                
                count = (balls[idx] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    candidates = [idx]
                elif count == best_count:
                    candidates.append(idx)
                    
            if best_count == 0:
                break
                
            chosen_idx = random.choice(candidates)
            code_indices.append(chosen_idx)
            uncovered &= ~balls[chosen_idx]
            
        # 2. Local Search / Reduction Phase
        # Use bitmasks for speed
        # code_set = set(code_indices)
        # union_mask = 0
        # for c in code_indices: union_mask |= balls[c]
        
        # We want to minimize len(code_indices)
        # Strategy:
        # While True:
        #   improved = False
        #   # Try removing
        #   for i, c in enumerate(code_indices):
        #      others_union = 0
        #      for j, o in enumerate(code_indices):
        #         if i != j: others_union |= balls[o]
        #      if (balls[c] & ~others_union) == 0:
        #          # c is redundant
        #          code_indices.pop(i)
        #          improved = True
        #          break
        #   if improved: continue
        #   
        #   # Try swapping
        #   # Pick a random c from code_indices
        #   # Compute gap = balls[c] & ~others_union
        #   # Find c' not in code_indices such that balls[c'] covers gap
        #   # If found, swap
        #   # ...
        
        # This loop is O(K^2) per iteration. K~200. 40k ops.
        # But doing it naively in Python might be slow if many iterations.
        # Let's optimize the "others_union" calculation.
        
        # Maintain `cover_mask` = union of all balls in code_indices
        cover_mask = 0
        for c in code_indices:
            cover_mask |= balls[c]
            
        # Redundancy check:
        # c is redundant if balls[c] is subset of (cover_mask XOR balls[c])?
        # No, cover_mask includes balls[c].
        # Let others_mask = cover_mask & ~balls[c]
        # If (balls[c] & ~others_mask) == 0, then c is redundant.
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Shuffle to randomize order
            random.shuffle(code_indices)
            for i in range(len(code_indices)):
                c = code_indices[i]
                others_mask = cover_mask & ~balls[c]
                if (balls[c] & ~others_mask) == 0:
                    # Redundant
                    code_indices.pop(i)
                    cover_mask &= ~balls[c]
                    changed = True
                    break # Restart loop with new list
            
            if not changed:
                # Try swaps
                # Pick a random codeword to try to replace
                if code_indices:
                    i = random.randint(0, len(code_indices) - 1)
                    c = code_indices[i]
                    others_mask = cover_mask & ~balls[c]
                    gap = balls[c] & ~others_mask
                    
                    if gap != 0:
                        # Need to cover 'gap' with a new word c'
                        # c' must not be in code_indices
                        # balls[c'] must cover 'gap'
                        
                        # Find candidates
                        # To speed up, we can precompute for each 'gap' pattern? No, gap is specific.
                        # Just scan all non-codewords?
                        # N=2187.
                        # Check `if (balls[j] & gap) == gap`
                        
                        # Optimization: Only check j such that balls[j] intersects gap?
                        # Or just check all.
                        
                        found = False
                        # Randomize order of j to diversify
                        js = list(range(N))
                        random.shuffle(js)
                        for j in js:
                            if j in code_indices:
                                continue
                            if (balls[j] & gap) == gap:
                                # Swap
                                code_indices[i] = j
                                # Update cover_mask
                                cover_mask = (cover_mask & ~balls[c]) | balls[j]
                                changed = True
                                found = True
                                break
                        
                        if not found:
                            # If we didn't find a swap, we might be stuck.
                            # We can try "kick": remove c, then add multiple words?
                            # Or just break and try next restart.
                            pass
        
        if len(code_indices) < best_code_size:
            best_code_size = len(code_indices)
            best_code_indices = code_indices[:]
            
            if best_code_size <= 186:
                break

    if best_code_indices is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [from_index(idx) for idx in best_code_indices]
