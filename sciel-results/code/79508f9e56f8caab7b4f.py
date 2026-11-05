"""Divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val, sorted."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_target = 128
    smooth = _generate_smooth(max_target)
    n_smooth = len(smooth)
    
    # Precompute compatibility mask: comp[i] is a bitmask of indices j 
    # such that smooth[i] and smooth[j] are incomparable (antichain pair)
    # Note: We only care about i != j.
    comp = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            # Check if smooth[i] and smooth[j] are incomparable
            if smooth[i] % smooth[j] != 0 and smooth[j] % smooth[i] != 0:
                # They are incomparable
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)
    
    # For each target, find minimum number of terms
    answers = [None] * max_target
    
    # Precompute candidates for each target as a bitmask
    # cand_mask[n] is a bitmask of indices i such that smooth[i] <= n
    cand_mask = [0] * (max_target + 1)
    for n in range(1, max_target + 1):
        mask = 0
        for i in range(n_smooth):
            if smooth[i] <= n:
                mask |= (1 << i)
            else:
                break # smooth is sorted
        cand_mask[n] = mask

    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    for n in range(1, max_target + 1):
        if time.time() > deadline:
            break
        if answers[n - 1] is not None:
            continue
            
        target_mask = cand_mask[n]
        if target_mask == 0:
            continue # No candidates, shouldn't happen for n>=1
            
        found = False
        # Try k from 1 upward
        for k in range(1, 31):
            if time.time() > deadline:
                break
            if found:
                break
            
            # DFS with bitmask pruning
            # State: (pos, remaining_sum, chosen_mask, current_allowed_mask)
            # We iterate through bits in current_allowed_mask
            
            result = None
            
            def dfs(remaining_sum, chosen_mask, allowed_mask, chosen_list):
                nonlocal result
                if time.time() > deadline:
                    return
                if result is not None:
                    return
                
                # Check if we have enough elements left in allowed_mask
                # We need (k - len(chosen_list)) more elements
                needed = k - len(chosen_list)
                if popcount(allowed_mask) < needed:
                    return
                
                if needed == 0:
                    if remaining_sum == 0:
                        result = list(chosen_list)
                    return
                
                # Pruning: Can we reach remaining_sum with 'needed' elements from allowed_mask?
                # Max possible sum: take the largest 'needed' elements from allowed_mask
                # Min possible sum: take the smallest 'needed' elements from allowed_mask
                
                # Extract indices from allowed_mask
                # Optimization: We can precompute sorted lists of indices for masks? 
                # No, masks change. 
                # Simple check: Sum of all elements in allowed_mask < remaining_sum?
                # That's expensive to compute every time.
                
                # Better: If the largest possible sum of 'needed' elements < remaining_sum, prune.
                # If the smallest possible sum of 'needed' elements > remaining_sum, prune.
                
                # Let's get the indices
                indices = []
                m = allowed_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    indices.append(idx)
                    m ^= lsb
                
                if len(indices) < needed:
                    return
                
                # Sort indices by value to get min/max sums
                # Since smooth is sorted, indices are sorted by value.
                # But allowed_mask might not be contiguous.
                # However, we can just take the first 'needed' and last 'needed' from the sorted list of indices?
                # Wait, indices are extracted in increasing order of bit position (which is value order).
                # So indices[0] is smallest, indices[-1] is largest.
                
                min_sum = sum(smooth[i] for i in indices[:needed])
                if min_sum > remaining_sum:
                    return
                
                max_sum = sum(smooth[i] for i in indices[-needed:])
                if max_sum < remaining_sum:
                    return
                
                # Iterate through candidates in allowed_mask
                # To find solution fast, maybe try larger numbers first? Or smaller?
                # For sum problems, order doesn't strictly matter for completeness, but for finding *a* solution, 
                # sometimes larger first helps hit the sum faster? Or smaller first to avoid overshooting?
                # Let's stick to increasing order for determinism, or maybe decreasing to prune "sum too small" faster?
                # Actually, if we pick a large number, remaining_sum drops fast.
                
                # Let's iterate through bits in allowed_mask
                m = allowed_mask
                while m:
                    if time.time() > deadline:
                        return
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    val = smooth[idx]
                    
                    if val <= remaining_sum:
                        # Pick idx
                        new_chosen_mask = chosen_mask | lsb
                        # New allowed mask: must be subset of allowed_mask without idx
                        # AND must be compatible with idx
                        new_allowed = (allowed_mask & ~lsb) & comp[idx]
                        
                        chosen_list.append(val)
                        dfs(remaining_sum - val, new_chosen_mask, new_allowed, chosen_list)
                        chosen_list.pop()
                        
                        if result is not None:
                            return
                    
                    m ^= lsb
            
            dfs(n, 0, target_mask, [])
            
            if result is not None:
                answers[n - 1] = result
                found = True
    
    # Fallback for any missing targets
    for n in range(1, max_target + 1):
        if answers[n - 1] is None:
            # This should be rare if the search is complete.
            # Simple brute force fallback
            from itertools import combinations
            cands = [x for x in smooth if x <= n]
            for k in range(1, len(cands) + 1):
                for terms in combinations(cands, k):
                    if sum(terms) == n:
                        # Check antichain
                        valid = True
                        for i in range(len(terms)):
                            for j in range(i+1, len(terms)):
                                if terms[i] % terms[j] == 0 or terms[j] % terms[i] == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            answers[n - 1] = list(terms)
                            break
                if answers[n - 1] is not None:
                    break
            if answers[n - 1] is None:
                # Should not happen
                answers[n - 1] = [n] if n in smooth else [1] # Invalid placeholder

    return answers
# EVOLVE-BLOCK-END
