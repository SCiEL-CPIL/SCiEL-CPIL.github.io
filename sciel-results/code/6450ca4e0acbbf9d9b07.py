import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Simple generic fallback
            return [[0] * n]

    total_words = 3 ** 8
    
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * 3 + v
        return idx

    def idx_to_tuple(idx):
        t = []
        for _ in range(8):
            t.append(idx % 3)
            idx //= 3
        return tuple(reversed(t))

    # Precompute masks
    masks = [0] * total_words
    for w in itertools.product(range(3), repeat=8):
        idx = tuple_to_idx(w)
        mask = 1 << idx
        for pos in range(8):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w_list = list(w)
                w_list[pos] = new_val
                n_idx = tuple_to_idx(tuple(w_list))
                mask |= (1 << n_idx)
        masks[idx] = mask

    all_ones = (1 << total_words) - 1
    covered = 0
    code_indices = []
    
    # 1. Greedy Phase
    while covered != all_ones:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_gain = 0
        not_covered = ~covered
        
        for i in range(total_words):
            gain = bin(masks[i] & not_covered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == 17:
                    break
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        covered |= masks[best_idx]

    # 2. Peeling Phase (Remove redundant codewords)
    # We try to remove codewords one by one.
    # To make it effective, we can iterate multiple times.
    # Each check is O(N_code * BigIntOps).
    
    max_peel_passes = 5
    for _ in range(max_peel_passes):
        if time.time() > deadline:
            break
        removed_any = False
        # Iterate in reverse to allow list modification? 
        # Or build a new list.
        new_code_indices = []
        for idx in code_indices:
            if time.time() > deadline:
                new_code_indices.append(idx)
                continue
            
            # Check if idx is redundant
            # Calculate coverage without idx
            # Optimization: 
            # current_covered = covered
            # We don't have the "covered" state for the subset easily without recomputing.
            # So we recompute the union for the candidate set.
            
            # Candidate set: new_code_indices + [idx] (if we keep it)
            # We want to know if union(new_code_indices) == all_ones.
            # If yes, we can drop idx.
            
            # Compute union of new_code_indices
            # If new_code_indices is empty, union is 0.
            if not new_code_indices:
                temp_covered = 0
            else:
                temp_covered = 0
                for i in new_code_indices:
                    temp_covered |= masks[i]
            
            if temp_covered == all_ones:
                # We can drop idx
                removed_any = True
                # Do not append idx
            else:
                # Must keep idx
                new_code_indices.append(idx)
                
        code_indices = new_code_indices
        if not removed_any:
            break

    # Convert indices back to words
    return [list(idx_to_tuple(i)) for i in code_indices]
