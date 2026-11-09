"""Direct construction exploiting divisibility structure of the LCM constraint.
Combined with small/large decomposition and submodular-aware local search."""
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    # Index mapping: index i -> value numbers[i] = i + 2

    # Precompute conflict masks using bit operations
    # conflict_mask[i] has bit j set if lcm(numbers[i], numbers[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    weights = [1.0 / numbers[i] for i in range(n)]

    def reciprocal_sum_from_mask(mask):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total

    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0

    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if conflict_mask[i] & (1 << j):
                            return False
        return True

    # === SMALL/LARGE DECOMPOSITION ===
    # Key insight: numbers > 128 are "large". For any two distinct odd numbers a, b > 128:
    #   lcm(a,b) = a*b/gcd(a,b). Since a,b > 128, a*b > 16384. gcd(a,b) <= min(a,b)/2 (if both even)
    #   Actually for odd numbers, gcd is odd. The minimum lcm for distinct odds > 128:
    #   a=129, b=135: gcd=9, lcm=129*135/9=1935 > 256. Safe.
    #   For even numbers > 128: a=130, b=132: gcd=2, lcm=130*132/2=8580 > 256. 
    #   Actually we need to be more careful. Let's check: can two numbers > 128 have lcm <= 256?
    #   lcm(a,b) <= 256 with a,b > 128 means a*b/gcd(a,b) <= 256, so a*b <= 256*gcd(a,b).
    #   Since a,b > 128, a*b > 16384. So we need 256*gcd(a,b) > 16384, i.e., gcd(a,b) > 64.
    #   So if gcd(a,b) > 64 and a,b > 128, lcm could be <= 256.
    #   Example: a=128, b=192: gcd=64, lcm=128*192/64=384 > 256. 
    #   a=129, b=258: but 258 > 256. 
    #   Let's just use the precomputed conflict masks.
    
    # Partition: small = [2, 128], large = [129, 256]
    small_indices = [i for i in range(n) if numbers[i] <= 128]
    large_indices = [i for i in range(n) if numbers[i] > 128]
    
    # For the large set, check which pairs conflict
    # If two large numbers don't conflict, they can both be included
    # The key structure: including a small number may block some large numbers
    
    # === DIRECT CONSTRUCTION (Strategy A: include 2) ===
    # If 2 is in S, no other even number can be (lcm(2, x) = x for even x, and x <= 256 means conflict)
    # All odd numbers > 128 are pairwise compatible (verified above)
    # Also, odd numbers <= 128 that don't conflict with 2: gcd(2, odd)=1, lcm=2*odd. 
    #   lcm(2, odd) = 2*odd. Need 2*odd > 256, so odd > 128. 
    #   So with 2 in S, we can only include odd numbers > 128.
    
    selected_mask_a = 0
    for i in range(n):
        val = numbers[i]
        if val == 2:
            selected_mask_a |= (1 << i)
        elif val % 2 == 1 and val > 128:
            selected_mask_a |= (1 << i)
    
    best_mask = selected_mask_a
    best_sum = reciprocal_sum_from_mask(selected_mask_a)
    
    assert is_valid(best_mask), "Direct construction A is invalid!"

    # === STRATEGY B: Exclude 2, use small/large decomposition ===
    # Without 2, we can include even numbers. 
    # The problem reduces to: choose a subset of small numbers, then include all compatible large numbers.
    # For a given subset S_small of small numbers, the compatible large numbers are those 
    # that don't conflict with any element in S_small.
    # Since large-large conflicts are rare (need gcd > 64), most large numbers will be compatible.
    
    # Let's build: for each small number, which large numbers does it block?
    small_blocks_large = {}
    for si in small_indices:
        blocked = 0
        for li in large_indices:
            if conflict_mask[si] & (1 << li):
                blocked |= (1 << li)
        small_blocks_large[si] = blocked
    
    # For the large set, build conflict among themselves
    large_conflict = {}
    for li in large_indices:
        large_conflict[li] = conflict_mask[li] & (sum(1 << x for x in large_indices))
    
    # Greedy on small numbers: pick small numbers that maximize (weight - cost of blocked large numbers)
    # But this is complex. Let's do a simpler approach:
    # Start with all large numbers that are mutually compatible, then add small numbers greedily.
    
    # First, find maximum weight independent set on large numbers (they're mostly compatible)
    # Greedy: sort large by weight descending, add if compatible with all selected large
    large_selected = 0
    for li in sorted(large_indices, key=lambda i: -weights[i]):
        if can_add(li, large_selected):
            large_selected |= (1 << li)
    
    # Now try adding small numbers
    strategy_b_mask = large_selected
    for si in sorted(small_indices, key=lambda i: -weights[i]):
        if can_add(si, strategy_b_mask):
            strategy_b_mask |= (1 << si)
    
    sum_b = reciprocal_sum_from_mask(strategy_b_mask)
    if sum_b > best_sum:
        best_sum = sum_b
        best_mask = strategy_b_mask

    # === STRATEGY C: Greedy by weight (global) ===
    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    mask_c = 0
    for i in order_w:
        if can_add(i, mask_c):
            mask_c |= (1 << i)
    sum_c = reciprocal_sum_from_mask(mask_c)
    if sum_c > best_sum:
        best_sum = sum_c
        best_mask = mask_c

    # === LOCAL SEARCH with small/large awareness ===
    # The key move: removing a small number may unblock multiple large numbers
    # So we try: remove one small number, re-greedy on large numbers
    
    def local_search_small_large(mask, deadline):
        """Local search that exploits small/large structure."""
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iterations = 0
        
        while improved and elapsed() < deadline:
            improved = False
            iterations += 1
            
            # Try removing each small number and re-optimizing large
            for si in small_indices:
                if not (current_mask & (1 << si)):
                    continue
                
                mask_no_si = current_mask & ~(1 << si)
                
                # Re-greedy on large numbers given the remaining small numbers
                # First, determine which large numbers are blocked by remaining smalls
                remaining_small = mask_no_si & (sum(1 << x for x in small_indices))
                
                # Find compatible large numbers
                blocked_large = 0
                for si2 in small_indices:
                    if remaining_small & (1 << si2):
                        blocked_large |= small_blocks_large[si2]
                
                # Greedy on unblocked large numbers
                large_new = 0
                for li in sorted(large_indices, key=lambda i: -weights[i]):
                    if blocked_large & (1 << li):
                        continue
                    if can_add(li, large_new | remaining_small):
                        large_new |= (1 << li)
                
                new_mask = remaining_small | large_new
                new_sum = reciprocal_sum_from_mask(new_mask)
                
                if new_sum > current_sum + 1e-12:
                    current_mask = new_mask
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try adding a large number that was previously blocked
            if not improved:
                for li in large_indices:
                    if not (current_mask & (1 << li)):
                        if can_add(li, current_mask):
                            new_mask = current_mask | (1 << li)
                            new_sum = reciprocal_sum_from_mask(new_mask)
                            if new_sum > current_sum + 1e-12:
                                current_mask = new_mask
                                current_sum = new_sum
                                improved = True
                                break
            
            # Try single swaps: remove one, add another
            if not improved:
                for i in range(n):
                    if not (current_mask & (1 << i)):
                        continue
                    for j in range(n):
                        if current_mask & (1 << j):
                            continue
                        mask_swap = (current_mask & ~(1 << j)) | (1 << i)
                        if is_valid(mask_swap):
                            new_sum = reciprocal_sum_from_mask(mask_swap)
                            if new_sum > current_sum + 1e-12:
                                current_mask = mask_swap
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        
        return current_mask, current_sum
    
    if elapsed() < DEADLINE:
        improved_mask, improved_sum = local_search_small_large(best_mask, DEADLINE)
        if improved_sum > best_sum:
            best_sum = improved_sum
            best_mask = improved_mask

    # === ADDITIONAL: Try anchor-based greedy for each anchor ===
    if elapsed() < DEADLINE - 5:
        # Try anchoring on each small number
        for anchor in small_indices:
            if elapsed() > DEADLINE - 2:
                break
            anchor_mask = 1 << anchor
            # Greedy completion: add highest weight compatible numbers
            for i in sorted(range(n), key=lambda x: -weights[x]):
                if not (anchor_mask & (1 << i)):
                    if can_add(i, anchor_mask):
                        anchor_mask |= (1 << i)
            
            anchor_sum = reciprocal_sum_from_mask(anchor_mask)
            if anchor_sum > best_sum:
                best_sum = anchor_sum
                best_mask = anchor_mask

    # === FINAL: Ensure validity and convert to result ===
    if not is_valid(best_mask):
        # Fallback to direct construction
        best_mask = selected_mask_a
        best_sum = reciprocal_sum_from_mask(best_mask)

    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()

    if not result:
        result = [2]

    return result
