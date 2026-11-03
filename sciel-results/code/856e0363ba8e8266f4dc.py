import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Handle non-standard cases gracefully, though task specifies 3, 7, 1
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Return a trivial covering (all words) if parameters differ
        # This ensures the function returns a valid covering code
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # 1. Pre-computation
    N = q ** n  # 2187
    all_mask = (1 << N) - 1
    
    # Mapping functions
    def word_to_idx(w):
        idx = 0
        for d in range(n - 1, -1, -1):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n - 1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Pre-compute balls (coverage sets) for each word
    balls = [0] * N
    # We generate words on the fly to save memory/time in precomp? 
    # Actually, we need to iterate all words to compute their balls.
    for i in range(N):
        w_i = idx_to_word(i)
        mask = 1 << i
        # Distance 1 neighbors
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                w_new = w_i[:]
                w_new[pos] = val
                j = word_to_idx(w_new)
                mask |= (1 << j)
        balls[i] = mask

    best_code = None
    best_size = N  # Worst case: all words are codewords

    # 2. Search Loop
    while time.time() < deadline:
        uncovered = all_mask
        selected = []
        
        # Greedy Selection with Randomized Tie-Breaking
        while uncovered:
            best_idx = -1
            best_score = -1.0
            
            # Iterate through all possible codewords
            # Optimization: We can skip indices already in selected? 
            # Not strictly necessary as their score will be low, but 
            # maintaining a set of selected is O(1) check.
            # However, N is small.
            
            for i in range(N):
                # Calculate how many new words this codeword covers
                new_coverage = (uncovered & balls[i]).bit_count()
                
                # Add noise to break ties and explore different solutions
                # The magnitude of noise can be tuned. 
                # If new_coverage is 0, it won't be picked unless all are 0 (which means uncovered is empty, loop ends).
                score = new_coverage + random.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # 3. Cleanup: Remove redundant codewords
        # A codeword is redundant if it doesn't cover any word that is not covered by others.
        # We iterate and try to remove each codeword.
        # To be efficient, we can do a single pass, but repeated passes ensure minimality.
        # Given |S| ~ 180, O(|S|^2) is fast.
        
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                # Compute coverage of all other selected codewords
                others_mask = 0
                for s in selected:
                    if s != c:
                        others_mask |= balls[s]
                
                # Check if c has any unique coverage
                unique_coverage = balls[c] & ~others_mask
                if unique_coverage == 0:
                    changed = True
                    continue # Skip adding c, it's redundant
                else:
                    new_selected.append(c)
            
            if changed:
                selected = new_selected
            else:
                break
                
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = [idx_to_word(i) for i in selected]
            
            # If we reach a very low number, we can keep searching or break early if we hit theoretical bounds?
            # Lower bound is 156. If we hit 156, we can break.
            if best_size <= 156:
                break

    return best_code
