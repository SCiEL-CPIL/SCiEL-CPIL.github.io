import time
import random
import numpy as np

def construct(q, n, radius):
    # Only handling the specific case q=3, n=6, radius=1 for optimal performance
    # but writing generally enough to not crash if called otherwise (though task implies fixed)
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other cases
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    # Precompute masks
    # word index -> bitmask of covered words
    masks = [0] * total_words
    
    # Helper to convert index to base-q digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits
    
    # Helper to convert base-q digits to index
    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    # Precompute neighbors
    for w_idx in range(total_words):
        d = idx_to_digits(w_idx)
        mask = 1 << w_idx
        # Distance 1 neighbors
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                d[pos] = new_val
                n_idx = digits_to_idx(d)
                mask |= (1 << n_idx)
            d[pos] = orig_val
        masks[w_idx] = mask

    # Greedy Set Cover
    uncovered = full_mask
    code = []
    
    # To speed up greedy, we can just iterate all words each time
    # 729 words, ~75 iterations. 729*75 popcounts is fast.
    while uncovered:
        best_w = -1
        best_covered = 0
        for w in range(total_words):
            covered = bin(uncovered & masks[w]).count('1')
            if covered > best_covered:
                best_covered = covered
                best_w = w
        code.append(best_w)
        uncovered &= ~masks[best_w]

    # Prune Function
    def prune(code_list):
        code_list = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                # Check if code_list[i] is redundant
                union = 0
                for j in range(len(code_list)):
                    if j != i:
                        union |= masks[code_list[j]]
                if union == full_mask:
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    best_code = prune(code)
    
    # Local Search
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Randomize for local search diversity?
    # We will try to improve best_code
    
    while time.time() < deadline:
        current_code = list(best_code)
        
        # Try a few random swap attempts
        # If we find a strictly better code, update best_code
        # If we find an equal code, keep it as current for further exploration?
        # Usually, just exploring from the best is safer.
        
        # Let's try to find a swap that allows a reduction
        # Pick a random codeword to remove
        if not current_code:
            break
            
        idx_to_remove = random.randint(0, len(current_code) - 1)
        c_old = current_code[idx_to_remove]
        
        # Find unique words covered by c_old
        unique_mask = masks[c_old]
        for c in current_code:
            if c != c_old:
                unique_mask &= ~masks[c]
                
        if unique_mask == 0:
            # It was redundant, prune should have caught it, but just in case
            continue
            
        # Find a replacement w
        # w must cover unique_mask
        # And w should not be in current_code (if it is, we are just swapping for nothing or removing)
        # Actually if w is in current_code, we are just removing c_old (if w covers unique_mask, 
        # then c_old is redundant given w is already there? No, w is already there, so c_old covers 
        # unique_mask, but w is already covering those? 
        # If w is in current_code, and w covers unique_mask, then c_old is redundant. 
        # So we should have pruned it. So w must NOT be in current_code.)
        
        candidates = []
        for w in range(total_words):
            if w in current_code:
                continue
            if (unique_mask & ~masks[w]) == 0:
                # w covers all unique words
                # Score w by how much extra it covers (to help future prunes)
                # Extra = bits in masks[w] not covered by (current_code \ {c_old})
                union_others = 0
                for c in current_code:
                    if c != c_old:
                        union_others |= masks[c]
                extra = bin(masks[w] & ~union_others).count('1')
                candidates.append((extra, w))
                
        if not candidates:
            continue
            
        # Pick the best candidate (most extra coverage)
        # Or random? Best is usually better.
        candidates.sort(reverse=True)
        best_extra, best_w = candidates[0]
        
        # Perform swap
        new_code = current_code.copy()
        new_code[idx_to_remove] = best_w
        new_code = prune(new_code)
        
        if len(new_code) < len(best_code):
            best_code = new_code
        # If equal, we could set current_code = new_code to explore, 
        # but restarting from best_code is simpler and avoids getting stuck in local minima 
        # of the "swap" graph that don't lead to reduction.
        
    # Convert indices back to lists of digits
    result = []
    for idx in best_code:
        d = idx_to_digits(idx)
        result.append(d)
        
    return result
