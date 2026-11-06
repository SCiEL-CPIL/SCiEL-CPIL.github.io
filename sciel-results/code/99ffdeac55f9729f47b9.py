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

def _precompute_minimal_representations():
    """
    Pre-compute minimal antichain representations for all targets 1..128.
    Uses bitmask-based antichain checking for efficiency.
    """
    start_time = time.time()
    DEADLINE = start_time + 75.0  # Safety margin before 90s kill
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Pre-compute incompatibility bitmask: inc[i] has bit j set if i and j are incompatible (one divides the other)
    inc = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                inc[i] |= (1 << j)
    
    def is_antichain(indices):
        """Check if all indices form an antichain."""
        for i in indices:
            for j in indices:
                if i != j and (inc[i] >> j) & 1:
                    return False
        return True
    
    def can_add(chosen_mask, idx):
        """Check if idx can be added to the set represented by chosen_mask."""
        for i in range(n_smooth):
            if (chosen_mask >> i) & 1:
                if (inc[i] >> idx) & 1:
                    return False
        return True
    
    # For each target, find minimal antichain representation
    answers = [None] * 129
    
    # Try 1-term representations
    for n in range(1, 129):
        if n in smooth_set:
            answers[n] = [n]
    
    # Try 2-term representations
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if (inc[i] >> j) & 1:
                continue
            s = si + smooth[j]
            if s <= 128 and answers[s] is None:
                answers[s] = [si, smooth[j]]
    
    # Try 3-term representations
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if (inc[i] >> j) & 1:
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if (inc[i] >> k) & 1 or (inc[j] >> k) & 1:
                    continue
                s = sij + smooth[k]
                if s <= 128 and answers[s] is None:
                    answers[s] = [si, smooth[j], smooth[k]]
    
    # Try 4-term representations
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if (inc[i] >> j) & 1:
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if (inc[i] >> k) & 1 or (inc[j] >> k) & 1:
                    continue
                sijk = sij + smooth[k]
                if sijk > 128:
                    continue
                for l in range(k + 1, n_smooth):
                    if (inc[i] >> l) & 1 or (inc[j] >> l) & 1 or (inc[k] >> l) & 1:
                        continue
                    s = sijk + smooth[l]
                    if s <= 128 and answers[s] is None:
                        answers[s] = [si, smooth[j], smooth[k], smooth[l]]
    
    # For remaining targets, use DFS to find representations with 5+ terms
    # Use iterative deepening DFS with bitmask for efficiency
    for n in range(1, 129):
        if answers[n] is not None:
            continue
        if time.time() > DEADLINE:
            break
        
        # DFS for 5+ terms using bitmask
        for count in range(5, 10):
            result = None
            # Use recursive DFS with bitmask
            def dfs(start_idx, chosen_mask, cur_sum, chosen_list):
                nonlocal result
                if result is not None:
                    return
                if len(chosen_list) == count:
                    if cur_sum == n:
                        result = list(chosen_list)
                    return
                
                remaining = count - len(chosen_list)
                for t in range(start_idx, n_smooth):
                    new_sum = cur_sum + smooth[t]
                    if new_sum > n:
                        break
                    
                    # Check antichain with all chosen using bitmask
                    valid = True
                    for c in range(n_smooth):
                        if (chosen_mask >> c) & 1:
                            if (inc[c] >> t) & 1:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Pruning: can we still reach n?
                    if remaining - 1 > 0:
                        # Minimum possible sum of remaining elements
                        min_possible = 0
                        for m in range(remaining - 1):
                            if t + 1 + m < n_smooth:
                                min_possible += smooth[t + 1 + m]
                            else:
                                min_possible = n + 1
                                break
                        if new_sum + min_possible > n:
                            continue
                    
                    new_mask = chosen_mask | (1 << t)
                    dfs(t + 1, new_mask, new_sum, chosen_list + [smooth[t]])
            
            dfs(0, 0, 0, [])
            
            if result is not None:
                answers[n] = result
                break
            if time.time() > DEADLINE:
                break
    
    return answers

# Pre-compute the lookup table once at module load time
_MINIMAL_REPRESENTATIONS = _precompute_minimal_representations()

def construct():
    """
    Return 128 rows corresponding to targets 1..128.
    Each row contains 1..30 distinct positive 2^a*3^b*5^c terms summing to its target,
    forming a divisibility antichain.
    """
    return _MINIMAL_REPRESENTATIONS[1:129]
# EVOLVE-BLOCK-END
