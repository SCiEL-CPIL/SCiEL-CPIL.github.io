import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0
    
    N = q ** n
    
    # If N is very large, this heuristic might be too slow.
    # But for the specific task N=729.
    
    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute ball bitsets
    # balls[i] is an integer bitmask of words covered by words[i]
    balls = [0] * N
    
    # Helper to get index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute distances or just generate balls
    # For efficiency, if radius is 1, we can use the neighbor generation trick.
    # Otherwise, compute Hamming distance.
    
    if radius == 1:
        for i, w in enumerate(words):
            mask = 1 << i
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = w.copy()
                    new_w[pos] = new_val
                    j = word_to_idx(new_w)
                    mask |= (1 << j)
            balls[i] = mask
    else:
        # Generic Hamming distance check
        # This is O(N^2 * n)
        for i in range(N):
            mask = 0
            wi = words[i]
            for j in range(N):
                wj = words[j]
                dist = sum(1 for k in range(n) if wi[k] != wj[k])
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            
    all_mask = (1 << N) - 1
    uncovered = all_mask
    selected = []
    
    # Greedy
    while uncovered != 0:
        if time.time() > deadline - 5:
            break
            
        best_idx = -1
        max_covered = -1
        
        for i in range(N):
            covered = balls[i] & uncovered
            cnt = bin(covered).count('1')
            if cnt > max_covered:
                max_covered = cnt
                best_idx = i
                
        if best_idx == -1 or max_covered == 0:
            break
            
        selected.append(best_idx)
        uncovered &= ~balls[best_idx]
        
    # Redundancy Removal
    def is_redundant(idx, sel_list):
        union_mask = 0
        for j in sel_list:
            if j != idx:
                union_mask |= balls[j]
        return (union_mask == all_mask)
        
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            if is_redundant(c, selected):
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
        if time.time() > deadline - 10:
            break
            
    # Convert back to list of lists
    result = [words[i] for i in selected]
    return result
