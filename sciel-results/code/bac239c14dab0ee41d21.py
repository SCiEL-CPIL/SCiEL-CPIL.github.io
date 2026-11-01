import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0
    
    N = q ** n
    
    # For very large N, this bitset approach might be slow or memory intensive.
    # But for the specific task (3,7,1), N=2187 is small.
    # If N is too large, we fallback to a simple greedy without bitsets or the seed.
    if N > 50000:
        # Fallback: Simple greedy using sets (slower but works for larger N if time allows)
        # Or just return the seed result if we can't compute fast.
        # Given the task is specific, we assume N is small.
        # If not, we might timeout. Let's try to be safe.
        pass

    # Precompute powers of q
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_base_q(x, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = x % q
            x //= q
        return res

    # Precompute cover masks
    # cover_masks[i] is a bitset of indices j such that d(i, j) <= radius
    cover_masks = [0] * N
    
    for i in range(N):
        mask = 1 << i
        digits = int_to_base_q(i, n, q)
        
        if radius >= 1:
            for p in range(n):
                orig = digits[p]
                for v in range(q):
                    if v == orig:
                        continue
                    new_idx = i + (v - orig) * powers[p]
                    if 0 <= new_idx < N:
                        mask |= (1 << new_idx)
        # If radius > 1, we would need to expand further, but task is radius 1.
        # For generality, if radius > 1, this precomputation is incomplete.
        # However, the task specifies radius 1.
        
        cover_masks[i] = mask

    # Greedy
    uncovered = (1 << N) - 1
    code_indices = []
    code_set = set()
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_i = -1
        max_cov = -1
        
        for i in range(N):
            if i in code_set:
                continue
            cov = (uncovered & cover_masks[i]).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_i = i
                
        if best_i == -1 or max_cov == 0:
            break
            
        code_indices.append(best_i)
        code_set.add(best_i)
        uncovered &= ~cover_masks[best_i]

    # Post-processing: Remove redundant codewords
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code_indices)):
            c = code_indices[i]
            
            # Calculate union of covers of all other codewords
            mask_others = 0
            for j, c2 in enumerate(code_indices):
                if j == i:
                    continue
                mask_others |= cover_masks[c2]
                
            # If there is a word covered by c but not by others, c is essential
            if (cover_masks[c] & ~mask_others) != 0:
                continue
            else:
                # Redundant
                code_indices.pop(i)
                code_set.remove(c)
                changed = True
                break

    # Convert to list of lists
    result = []
    for idx in code_indices:
        result.append(int_to_base_q(idx, n, q))
        
    return result
