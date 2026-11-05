import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Only handle the specific case q=3, n=7, radius=1 efficiently
    # Fallback to seed logic for other cases if needed, but task is specific
    if q != 3 or n != 7 or radius != 1:
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n  # 2187
    
    # Generate all words and map to indices
    words = []
    for i in range(total_words):
        w = []
        v = i
        for _ in range(n):
            w.append(v % q)
            v //= q
        words.append(tuple(w))
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute masks: masks[i] is a bitmask where bit j is 1 if word j is in ball of word i
    masks = [0] * total_words
    for i, w in enumerate(words):
        mask = 0
        mask |= (1 << word_to_idx[w])
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                mask |= (1 << word_to_idx[new_w])
        masks[i] = mask
    
    # Greedy Set Cover
    uncovered_mask = (1 << total_words) - 1
    selected_indices = []
    
    while uncovered_mask != 0:
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            overlap = masks[i] & uncovered_mask
            if overlap:
                count = overlap.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered_mask &= ~masks[best_idx]
    
    # Local search: try to remove redundant codewords
    def is_redundant(idx, selected_set, masks):
        # Check if removing idx still covers all words
        # A codeword idx is redundant if every word in its ball is also in the ball of some other selected codeword
        # More precisely: if for every word w in ball(idx), there exists some other selected idx' != idx such that w is in ball(idx')
        # But that's expensive. Instead: idx is redundant if the union of balls of other selected codewords still covers all of ball(idx)
        # Actually, we need: the union of balls of (selected_set - {idx}) still covers all 2187 words.
        # That's expensive. Cheaper: idx is redundant if every word in ball(idx) is covered by at least one other selected codeword.
        # But that's not sufficient - we need global coverage.
        # For now, use a simpler check: remove idx and see if coverage is still complete.
        # This is expensive but n=7 is small.
        
        # Actually, let's use the bitmask approach:
        # Compute union of masks of all selected except idx
        # Check if it equals (1 << total_words) - 1
        
        union_mask = 0
        for i in selected_set:
            if i != idx:
                union_mask |= masks[i]
        
        return (union_mask == (1 << total_words) - 1)
    
    # Try to remove redundants
    removed_any = True
    while removed_any and (time.time() - start_time < time_limit):
        removed_any = False
        selected_set = set(selected_indices)
        for idx in list(selected_indices):
            if is_redundant(idx, selected_set, masks):
                selected_indices.remove(idx)
                selected_set.remove(idx)
                removed_any = True
                break
    
    # Try swaps: replace one codeword with another that allows removing a different one
    # This is more complex; for now, just return the result after redundant removal
    
    result = [words[i] for i in selected_indices]
    return [list(w) for w in result]
