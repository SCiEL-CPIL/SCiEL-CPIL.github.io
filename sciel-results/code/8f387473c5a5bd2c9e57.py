"""Direct construction exploiting divisibility structure of the LCM constraint."""
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # Precompute conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
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

    # === DIRECT CONSTRUCTION ===
    # Key structural facts:
    # 1. If 2 in S, no other even can be (lcm(2,x)=x<=256 for even x)
    # 2. Odds > 128 are mutually compatible (lcm > 256 always)
    # 3. Evens don't conflict with odds > 128 (lcm(e,o) > 256 for even e>=2, odd o>128)
    # 4. So: include 2 + all odds>128, OR exclude 2 + select compatible evens + all odds>128

    # Build the direct solution: {2} union {odd numbers in (128, 256]}
    selected_mask = 0
    for i in range(n):
        val = numbers[i]
        if val == 2:
            selected_mask |= (1 << i)
        elif val % 2 == 1 and val > 128:
            selected_mask |= (1 << i)
    
    best_mask = selected_mask
    best_sum = reciprocal_sum_from_mask(selected_mask)
    
    # Verify validity
    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i+1, n):
                    if mask & (1 << j):
                        if conflict_mask[i] & (1 << j):
                            return False
        return True
    
    assert is_valid(best_mask), "Direct construction is invalid!"
    
    # === GREEDY FROM MULTIPLE STARTS ===
    def greedy(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask
    
    # Ordering 1: weight descending
    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    mask_w = greedy(order_w)
    sum_w = reciprocal_sum_from_mask(mask_w)
    if sum_w > best_sum:
        best_sum = sum_w
        best_mask = mask_w
    
    # Ordering 2: "no 2" strategy - exclude 2, prioritize evens by weight
    order_no2 = [i for i in range(n) if numbers[i] != 2]
    order_no2.sort(key=lambda i: -weights[i])
    mask_no2 = greedy(order_no2)
    sum_no2 = reciprocal_sum_from_mask(mask_no2)
    if sum_no2 > best_sum:
        best_sum = sum_no2
        best_mask = mask_no2
    
    # Ordering 3: "no 2" with evens first (to maximize even coverage)
    order_ev_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 0]
    order_odd_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 1]
    order_ev_first.sort(key=lambda i: -weights[i])
    order_odd_first.sort(key=lambda i: -weights[i])
    mask_mixed = greedy(order_ev_first + order_odd_first)
    sum_mixed = reciprocal_sum_from_mask(mask_mixed)
    if sum_mixed > best_sum:
        best_sum = sum_mixed
        best_mask = mask_mixed
    
    # === LOCAL SEARCH ===
    def try_additions(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)):
                if can_add(i, new_mask):
                    new_mask |= (1 << i)
        return new_mask
    
    def local_search(mask, max_iters=50):
        """Iterative local search with 1-swap and 2-swap moves."""
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1
            
            # Phase 1: Add all possible
            current_mask = try_additions(current_mask)
            current_sum = reciprocal_sum_from_mask(current_mask)
            
            # Phase 2: Try 1-removal + greedy fill
            for i in range(n):
                if elapsed() > DEADLINE:
                    break
                if not (current_mask & (1 << i)):
                    continue
                mask_no_i = current_mask & ~(1 << i)
                new_mask = try_additions(mask_no_i)
                new_s = reciprocal_sum_from_mask(new_mask)
                if new_s > current_sum + 1e-12:
                    current_mask = new_mask
                    current_sum = new_s
                    improved = True
                    break
            
            # Phase 3: Try 2-swap (remove one, add another)
            if not improved and iters < max_iters and elapsed() < DEADLINE:
                for i in range(n):
                    if elapsed() > DEADLINE:
                        break
                    if not (current_mask & (1 << i)):
                        continue
                    mask_no_i = current_mask & ~(1 << i)
                    # Try adding each non-selected element
                    best_j = -1
                    best_gain = 0.0
                    for j in range(n):
                        if (current_mask & (1 << j)):
                            continue
                        if can_add(j, mask_no_i):
                            gain = weights[j] - weights[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_j = j
                    if best_j >= 0 and best_gain > 1e-12:
                        new_mask = mask_no_i | (1 << best_j)
                        current_mask = new_mask
                        current_sum += best_gain
                        improved = True
                        break
        
        return current_mask, current_sum
    
    # Run local search from best candidate
    selected = best_mask
    selected, sel_sum = local_search(selected, max_iters=100)
    if sel_sum > best_sum:
        best_sum = sel_sum
        best_mask = selected
    
    # Also run local search from the "no 2" greedy result
    if mask_no2 != best_mask:
        sel2, sum2 = local_search(mask_no2, max_iters=50)
        if sum2 > best_sum:
            best_sum = sum2
            best_mask = sel2
    
    # Convert to result
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    
    if not result:
        result = [2]
    
    return result
