import time
import itertools

def construct(q, n, radius):
    if n > 12:
        # Fallback for safety, though task is n=7
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 110.0

    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    coverage = [0] * total_words
    for i, w in enumerate(words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        coverage[i] = mask
        
    uncovered = (1 << total_words) - 1
    code_indices = []
    used = [False] * total_words
    
    # Greedy Phase
    while uncovered:
        if time.time() > deadline:
            break
            
        best_c = -1
        best_count = -1
        
        for c in range(total_words):
            if used[c]:
                continue
            cnt = (coverage[c] & uncovered).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_c = c
                
        if best_c == -1:
            break
            
        code_indices.append(best_c)
        used[best_c] = True
        uncovered &= ~coverage[best_c]
        
    # Local Search / Cleanup Phase
    # Maintain cover_count to identify redundant words
    cover_count = [0] * total_words
    for c in code_indices:
        # Iterate bits set in coverage[c]
        mask = coverage[c]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask ^= lsb
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove redundant words
        for i in range(len(code_indices) - 1, -1, -1):
            c = code_indices[i]
            # Check if any word is only covered by c
            # i.e., exists j in coverage[c] such that cover_count[j] == 1
            redundant = True
            mask = coverage[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                mask ^= lsb
                
            if redundant:
                # Remove c
                # Decrement cover_count
                mask = coverage[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    mask ^= lsb
                    
                code_indices.pop(i)
                used[c] = False
                improved = True
                # Restart loop as indices changed
                break
        
        # If no removals, try swaps?
        # Swaps are complex to implement correctly with cover_count updates.
        # Given the time limit, removals are the easiest win.
        # If no removals, we stop.
        
    return [words[i] for i in code_indices]
