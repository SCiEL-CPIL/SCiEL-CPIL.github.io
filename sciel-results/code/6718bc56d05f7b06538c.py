"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute conflict: conflict[i] is set of j where smooth[i] divides smooth[j] or vice versa
    conflict = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                conflict[i].add(j)
    
    # Generate all antichains and record minimal size for each sum
    # This is the "inspiration" strategy: enumerate valid configurations
    # instead of searching for a specific sum each time.
    
    # best_for_sum[s] = (min_count, mask)
    best_for_sum = [None] * 129
    
    # We use a recursive generation of antichains.
    # State: (next_index, current_mask, blocked_mask)
    # To optimize, we don't store all antichains, just the best for each sum.
    
    def generate_and_record(start_idx, current_mask, blocked_mask):
        if time.time() > deadline:
            return
        
        # If current_mask is not empty, it's a valid antichain.
        if current_mask != 0:
            # Calculate sum and count
            s = 0
            cnt = 0
            temp_mask = current_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                s += smooth[idx]
                cnt += 1
                temp_mask ^= lsb
            
            if s <= 128:
                if best_for_sum[s] is None or cnt < best_for_sum[s][0]:
                    best_for_sum[s] = (cnt, current_mask)
        
        # Try adding more elements
        for i in range(start_idx, n_smooth):
            if time.time() > deadline:
                return
            
            # If i is blocked, skip
            if blocked_mask & (1 << i):
                continue
            
            # Choose i
            new_mask = current_mask | (1 << i)
            # Update blocked: add i and all its conflicts
            new_blocked = blocked_mask | (1 << i) | conflict[i]
            
            generate_and_record(i + 1, new_mask, new_blocked)
            
            # Note: We do NOT explicitly recurse for the "skip i" case here because
            # the loop naturally moves to i+1. The recursion structure "pick i, then pick from >i"
            # combined with the loop effectively covers all subsets.
            # Wait, this logic is slightly off for standard subset generation.
            # Standard pattern:
            # for i in range(start, n):
            #   if valid:
            #     pick i
            #     recurse(i+1, mask|i, blocked|conflicts)
            #   # implicit skip i is handled by continuing the loop
            # This correctly generates all antichains.
    
    generate_and_record(0, 0, 0)
    
    # Fallback for any target not covered (should be rare/impossible for 1..128 with smooth numbers)
    # If a target is smooth, it can be represented by itself (1 term).
    # The generation above should have captured single-element antichains.
    # Let's verify and fill any gaps.
    
    for n in range(1, 129):
        if best_for_sum[n] is None:
            # Check if n is in smooth
            if n in smooth:
                idx = smooth.index(n)
                mask = 1 << idx
                best_for_sum[n] = (1, mask)
            else:
                # If still None, we need a search fallback.
                # This should not happen if the antichain generation is complete.
                # But let's add a safety net using a quick search for any valid rep.
                # We'll use a simple backtracking for the specific target.
                res = _find_any_min(n, smooth, conflict)
                if res:
                    mask = 0
                    for idx in res:
                        mask |= (1 << idx)
                    best_for_sum[n] = (len(res), mask)
                else:
                    # Last resort: empty? No, must sum to n.
                    # This case implies no solution exists, but problem says "All targets require valid representations".
                    # So we assume a solution exists.
                    pass

    answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            cnt, mask = best_for_sum[n]
            terms = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                terms.append(smooth[idx])
                temp_mask ^= lsb
            answers.append(terms)
        else:
            # Should not happen
            answers.append([])
            
    return answers

def _find_any_min(target, smooth, conflict):
    """Fallback search for minimal antichain summing to target."""
    n = len(smooth)
    # Try k=1, 2, ...
    for k in range(1, 10):
        res = _search_k(target, k, 0, 0, set(), smooth, conflict, n)
        if res is not None:
            return res
    return None

def _search_k(target, k, start_idx, current_sum, chosen, smooth, conflict, n):
    if k == 0:
        if current_sum == target:
            return list(chosen)
        return None
    
    remaining = target - current_sum
    if remaining < k:
        return None
    
    for i in range(start_idx, n):
        val = smooth[i]
        if val > remaining:
            continue
        
        valid = True
        for c in chosen:
            if c in conflict[i]:
                valid = False
                break
        if not valid:
            continue
            
        new_chosen = chosen | {i}
        res = _search_k(target, k-1, i+1, current_sum+val, new_chosen, smooth, conflict, n)
        if res is not None:
            return res
    return None

# EVOLVE-BLOCK-END
