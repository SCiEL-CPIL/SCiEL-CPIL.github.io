"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Pre-compute divisibility: compat[i] = bitmask of indices j such that smooth[i] and smooth[j] are incomparable (neither divides the other)
    # Actually, we want: for each i, the set of j > i that are compatible with i
    # compat_mask[i] = bitmask where bit j is set if j > i and smooth[i] and smooth[j] form an antichain pair
    compat_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if b % a != 0 and a % b != 0:
                compat_mask[i] |= (1 << j)
    
    # For a set of chosen indices, the compatible candidates are those j such that:
    #   j > max(chosen) AND for all c in chosen, bit j is set in compat_mask[c]
    # We can precompute for each subset... but that's too many. Instead, use a running mask.
    
    # Precompute antichains by (size, sum) using efficient search
    # antichains_by_sum[size][sum] = tuple of indices (or None)
    MAX_SIZE = 10
    antichains_by_sum = [dict() for _ in range(MAX_SIZE + 1)]
    
    # For size 1
    for i in range(n_smooth):
        s = smooth[i]
        if s <= 128:
            if s not in antichains_by_sum[1]:
                antichains_by_sum[1][s] = (i,)
    
    # For sizes 2 through MAX_SIZE, use DFS with pruning
    # We'll do a global DFS that records the first antichain found for each (size, sum)
    # To be efficient, we process in order and stop early when all targets are covered for a given size
    
    # Track which sums are already covered for each size
    covered = [set() for _ in range(MAX_SIZE + 1)]
    for i in range(n_smooth):
        covered[1].add(smooth[i])
    
    # DFS to find antichains of a given size
    def find_antichains_of_size(size, target_sums_needed):
        """Find antichains of exactly 'size' elements, recording the first one for each sum in target_sums_needed."""
        if size == 0:
            return
        if size == 1:
            for i in range(n_smooth):
                s = smooth[i]
                if s in target_sums_needed and s not in covered[size]:
                    antichains_by_sum[size][s] = (i,)
                    covered[size].add(s)
            return
        
        # For size >= 2, use iterative deepening DFS
        # chosen: list of indices chosen so far
        # start: minimum index to consider next
        # current_sum: sum of chosen so far
        # compat_and: bitmask of indices that are compatible with ALL chosen elements
        def dfs(start, chosen, current_sum, compat_and):
            if time.time() > deadline:
                return False
            remaining = size - len(chosen)
            if remaining == 1:
                # Need to find one more element
                # Candidates: indices j >= start such that bit j is set in compat_and
                # and current_sum + smooth[j] is in target_sums_needed
                # Find the smallest such j
                # We can iterate through bits set in compat_and that are >= start
                mask = compat_and & ~((1 << start) - 1)
                while mask:
                    # Get lowest set bit
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    s = current_sum + smooth[j]
                    if s in target_sums_needed and s not in covered[size]:
                        antichains_by_sum[size][s] = tuple(chosen) + (j,)
                        covered[size].add(s)
                    mask &= mask - 1  # clear lowest set bit
                return len(target_sums_needed - covered[size]) == 0
            
            if remaining == 2:
                # Need two more elements j < k
                # Iterate over j >= start, then for each j, iterate over k > j with bit k in (compat_and & compat_mask[j])
                mask = compat_and & ~((1 << start) - 1)
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    # Now find k > j such that bit k is in (compat_and & compat_mask[j])
                    inner_mask = (compat_and & compat_mask[j]) & ~((1 << (j+1)) - 1)
                    while inner_mask:
                        lsb2 = inner_mask & (-inner_mask)
                        k = lsb2.bit_length() - 1
                        s = current_sum + smooth[j] + smooth[k]
                        if s in target_sums_needed and s not in covered[size]:
                            antichains_by_sum[size][s] = tuple(chosen) + (j, k)
                            covered[size].add(s)
                        inner_mask &= inner_mask - 1
                    mask &= mask - 1
                return len(target_sums_needed - covered[size]) == 0
            
            # General case: remaining >= 3
            # Pruning: minimum possible sum with 'remaining' more elements
            # The smallest 'remaining' elements from the candidate set
            mask = compat_and & ~((1 << start) - 1)
            # Count bits
            if mask.bit_count() < remaining:
                return False
            
            # Iterate over candidates for next element
            while mask:
                lsb = mask & (-mask)
                i = lsb.bit_length() - 1
                new_sum = current_sum + smooth[i]
                # Pruning: even if we pick the smallest (remaining-1) elements after i, 
                # can we still reach a target sum?
                # For now, just check new_sum <= 128
                if new_sum > 128:
                    mask &= mask - 1
                    continue
                
                new_compat = compat_and & compat_mask[i]
                # Remove bits <= i
                new_compat &= ~((1 << (i+1)) - 1)
                
                chosen.append(i)
                done = dfs(i + 1, chosen, new_sum, new_compat)
                chosen.pop()
                
                if done:
                    return True
                
                mask &= mask - 1
            
            return False
        
        # Start DFS
        target_sums = target_sums_needed - covered[size]
        if not target_sums:
            return
        dfs(0, [], 0, (1 << n_smooth) - 1)
    
    # Determine which sums need coverage for each size
    all_targets = set(range(1, 129))
    
    for size in range(2, MAX_SIZE + 1):
        # Sums not yet covered by smaller sizes
        already_covered = set()
        for s in range(1, size + 1):
            already_covered |= covered[s]
        needed = all_targets - already_covered
        if not needed:
            break
        if time.time() > deadline:
            break
        find_antichains_of_size(size, needed)
    
    # Now solve each target
    answers = [None] * 129
    
    total_terms = 0
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        answer = None
        
        # Try sizes 1 through MAX_SIZE
        for count in range(1, MAX_SIZE + 1):
            if n in antichains_by_sum[count]:
                idxs = antichains_by_sum[count][n]
                answer = [smooth[i] for i in idxs]
                break
        
        # Fallback: recursive search for larger sizes
        if answer is None:
            found = False
            for count in range(MAX_SIZE + 1, 21):
                if time.time() > deadline:
                    break
                
                # DFS with bitmask compatibility
                def dfs_fallback(start, chosen, current_sum, compat_and, remaining):
                    if time.time() > deadline:
                        return None
                    if remaining == 1:
                        mask = compat_and & ~((1 << start) - 1)
                        while mask:
                            lsb = mask & (-mask)
                            j = lsb.bit_length() - 1
                            if current_sum + smooth[j] == n:
                                return chosen + [j]
                            mask &= mask - 1
                        return None
                    
                    if remaining == 2:
                        mask = compat_and & ~((1 << start) - 1)
                        while mask:
                            lsb = mask & (-mask)
                            j = lsb.bit_length() - 1
                            inner_mask = (compat_and & compat_mask[j]) & ~((1 << (j+1)) - 1)
                            while inner_mask:
                                lsb2 = inner_mask & (-inner_mask)
                                k = lsb2.bit_length() - 1
                                if current_sum + smooth[j] + smooth[k] == n:
                                    return chosen + [j, k]
                                inner_mask &= inner_mask - 1
                            mask &= mask - 1
                        return None
                    
                    # General case
                    mask = compat_and & ~((1 << start) - 1)
                    if mask.bit_count() < remaining:
                        return None
                    
                    while mask:
                        lsb = mask & (-mask)
                        i = lsb.bit_length() - 1
                        new_sum = current_sum + smooth[i]
                        if new_sum > n:
                            mask &= mask - 1
                            continue
                        
                        new_compat = compat_and & compat_mask[i]
                        new_compat &= ~((1 << (i+1)) - 1)
                        
                        result = dfs_fallback(i + 1, chosen + [i], new_sum, new_compat, remaining - 1)
                        if result is not None:
                            return result
                        
                        mask &= mask - 1
                    
                    return None
                
                result = dfs_fallback(0, [], 0, (1 << n_smooth) - 1, count)
                if result is not None:
                    answer = [smooth[i] for i in result]
                    found = True
                    break
        
        if answer is None:
            # Last resort fallback
            if n in smooth_set:
                answer = [n]
            else:
                # Try to find any valid representation
                for count in range(1, 31):
                    def dfs_last(start, chosen, current_sum, compat_and, remaining):
                        if time.time() > deadline:
                            return None
                        if remaining == 0:
                            if current_sum == n:
                                return chosen
                            return None
                        if current_sum > n:
                            return None
                        mask = compat_and & ~((1 << start) - 1)
                        if mask.bit_count() < remaining:
                            return None
                        while mask:
                            lsb = mask & (-mask)
                            i = lsb.bit_length() - 1
                            new_sum = current_sum + smooth[i]
                            if new_sum > n:
                                mask &= mask - 1
                                continue
                            new_compat = compat_and & compat_mask[i]
                            new_compat &= ~((1 << (i+1)) - 1)
                            result = dfs_last(i + 1, chosen + [i], new_sum, new_compat, remaining - 1)
                            if result is not None:
                                return result
                            mask &= mask - 1
                        return None
                    result = dfs_last(0, [], 0, (1 << n_smooth) - 1, count)
                    if result is not None:
                        answer = [smooth[i] for i in result]
                        break
        
        if answer is not None:
            total_terms += len(answer)
        answers[n] = answer
    
    # Fill any remaining unfilled targets
    for n in range(1, 129):
        if answers[n] is None:
            for count in range(1, 31):
                if time.time() > deadline:
                    break
                def dfs_fill(start, chosen, current_sum, compat_and, remaining):
                    if time.time() > deadline:
                        return None
                    if remaining == 0:
                        if current_sum == n:
                            return chosen
                        return None
                    if current_sum > n:
                        return None
                    mask = compat_and & ~((1 << start) - 1)
                    if mask.bit_count() < remaining:
                        return None
                    while mask:
                        lsb = mask & (-mask)
                        i = lsb.bit_length() - 1
                        new_sum = current_sum + smooth[i]
                        if new_sum > n:
                            mask &= mask - 1
                            continue
                        new_compat = compat_and & compat_mask[i]
                        new_compat &= ~((1 << (i+1)) - 1)
                        result = dfs_fill(i + 1, chosen + [i], new_sum, new_compat, remaining - 1)
                        if result is not None:
                            return result
                        mask &= mask - 1
                    return None
                result = dfs_fill(0, [], 0, (1 << n_smooth) - 1, count)
                if result is not None:
                    answers[n] = [smooth[i] for i in result]
                    break
            else:
                # Absolute fallback - this should never happen
                answers[n] = [n] if n in smooth_set else [1, 2, 3]
    
    return answers[1:129]
# EVOLVE-BLOCK-END
