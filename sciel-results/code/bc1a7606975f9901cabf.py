import itertools
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbor_masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
        neighbor_masks[i] = mask

    # 1. Greedy Initial Solution
    current_uncovered = (1 << total_words) - 1
    code_indices = []
    while current_uncovered:
        best_idx = -1
        max_gain = 0
        for i in range(total_words):
            gain = (neighbor_masks[i] & current_uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1: break
        code_indices.append(best_idx)
        current_uncovered &= ~(neighbor_masks[best_idx])
        
    best_size = len(code_indices)
    best_code = code_indices[:]

    # 2. Backtracking Search (Limited Discrepancy)
    # We try to find a code of size < best_size
    sys.setrecursionlimit(10000)
    
    def search(uncovered_mask, depth, chosen):
        nonlocal best_size, best_code
        
        if not uncovered_mask:
            if depth < best_size:
                best_size = depth
                best_code = chosen[:]
            return
        
        if depth >= best_size:
            return
            
        # Lower Bound
        lb = (uncovered_mask.bit_count() + 14) // 15
        if depth + lb >= best_size:
            return
            
        # Calculate gains for all candidates
        gains = []
        for i in range(total_words):
            gain = (neighbor_masks[i] & uncovered_mask).bit_count()
            if gain > 0:
                gains.append((gain, i))
                
        if not gains:
            return
            
        gains.sort(reverse=True)
        max_gain = gains[0][0]
        
        # Branch on candidates with gain close to max
        # To keep it fast, only branch on top 2 or 3
        # If max_gain is small, branching is less critical
        candidates = []
        for gain, idx in gains:
            if gain >= max_gain - 1: # Allow 1 discrepancy
                candidates.append(idx)
            else:
                break # Since sorted, rest are smaller
                
        # If too many candidates, limit to top 3
        if len(candidates) > 3:
            candidates = candidates[:3]
            
        for idx in candidates:
            new_uncovered = uncovered_mask & ~neighbor_masks[idx]
            chosen.append(idx)
            search(new_uncovered, depth + 1, chosen)
            chosen.pop()
            
    search((1 << total_words) - 1, 0, [])
    
    return [words[i] for i in best_code]
