"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict graph using bitmasks for speed
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:  # lcm <= 256, conflict
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Reciprocal weights
    weights = [1.0 / numbers[i] for i in range(n)]
    
    # Partition into small (2-128) and large (129-256)
    small_indices = [i for i in range(n) if numbers[i] <= 128]
    large_indices = [i for i in range(n) if numbers[i] > 128]
    
    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(mask):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total
    
    def greedy_construct(order):
        selected_mask = 0
        for i in order:
            if can_add(i, selected_mask):
                selected_mask |= (1 << i)
        return selected_mask
    
    def fill_mask(mask):
        """Greedily add all non-conflicting nodes."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not (mask & (1 << i)) and can_add(i, mask):
                    mask |= (1 << i)
                    changed = True
        return mask
    
    # Compute adjusted weights: for each small number, add marginal gain from compatible large numbers
    # A small number s is "compatible" with large number L if lcm(s,L) > 256
    # The adjusted weight accounts for the fact that selecting s may block some large numbers
    adjusted_weights = list(weights)
    for s_idx in small_indices:
        s_val = numbers[s_idx]
        # Find large numbers that conflict with s
        conflicting_large = []
        for l_idx in large_indices:
            l_val = numbers[l_idx]
            g = gcd(s_val, l_val)
            l = s_val * l_val // g
            if l <= N:
                conflicting_large.append(l_idx)
        
        # The marginal cost of selecting s is the sum of weights of conflicting large numbers
        # that would otherwise be selected (approximation: all of them)
        if conflicting_large:
            marginal_cost = sum(weights[l_idx] for l_idx in conflicting_large)
            # Adjusted weight = base weight - marginal cost (but keep positive)
            adjusted_weights[s_idx] = max(0.0, weights[s_idx] - 0.5 * marginal_cost)
        else:
            # No conflicting large numbers, full weight
            adjusted_weights[s_idx] = weights[s_idx]
    
    # Try multiple greedy orderings using both original and adjusted weights
    best_mask = 0
    best_sum = 0.0
    
    def try_ordering(order, mask, sum_val):
        nonlocal best_mask, best_sum
        if sum_val > best_sum:
            best_sum = sum_val
            best_mask = mask
    
    # Ordering 1: by original weight descending
    order1 = list(range(n))
    order1.sort(key=lambda i: (-weights[i], bin(conflict_mask[i]).count('1')))
    mask1 = fill_mask(greedy_construct(order1))
    sum1 = reciprocal_sum_from_mask(mask1)
    try_ordering(order1, mask1, sum1)
    
    # Ordering 2: by adjusted weight descending (bipartite-aware)
    order2 = list(range(n))
    order2.sort(key=lambda i: (-adjusted_weights[i], -weights[i]))
    mask2 = fill_mask(greedy_construct(order2))
    sum2 = reciprocal_sum_from_mask(mask2)
    try_ordering(order2, mask2, sum2)
    
    # Ordering 3: weight / conflict_count
    cc = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order3 = list(range(n))
    order3.sort(key=lambda i: (-(weights[i] / (1 + cc[i])), -weights[i]))
    mask3 = fill_mask(greedy_construct(order3))
    sum3 = reciprocal_sum_from_mask(mask3)
    try_ordering(order3, mask3, sum3)
    
    # Ordering 4: adjusted weight / conflict_count
    order4 = list(range(n))
    order4.sort(key=lambda i: (-(adjusted_weights[i] / (1 + cc[i])), -weights[i]))
    mask4 = fill_mask(greedy_construct(order4))
    sum4 = reciprocal_sum_from_mask(mask4)
    try_ordering(order4, mask4, sum4)
    
    # Ordering 5: large numbers first, then small by adjusted weight
    order5 = large_indices + sorted(small_indices, key=lambda i: -adjusted_weights[i])
    mask5 = fill_mask(greedy_construct(order5))
    sum5 = reciprocal_sum_from_mask(mask5)
    try_ordering(order5, mask5, sum5)
    
    # Ordering 6: small numbers first by adjusted weight, then large
    order6 = sorted(small_indices, key=lambda i: -adjusted_weights[i]) + large_indices
    mask6 = fill_mask(greedy_construct(order6))
    sum6 = reciprocal_sum_from_mask(mask6)
    try_ordering(order6, mask6, sum6)
    
    # Random restarts with biased ordering
    random.seed(42)
    for _ in range(20):
        if elapsed() > DEADLINE - 15:
            break
        order_r = list(range(n))
        random.shuffle(order_r)
        # Bias: sort partially by adjusted weight
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            if random.random() < 0.75:
                if adjusted_weights[order_r[i]] < adjusted_weights[order_r[j]]:
                    order_r[i], order_r[j] = order_r[j], order_r[i]
        mask_r = fill_mask(greedy_construct(order_r))
        sum_r = reciprocal_sum_from_mask(mask_r)
        try_ordering(order_r, mask_r, sum_r)
    
    # Local search: 1-removal + fill
    def local_search_1swap(mask, max_iters=100):
        current_sum = reciprocal_sum_from_mask(mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                mask_without_i = mask & ~(1 << i)
                new_mask = fill_mask(mask_without_i)
                new_sum = reciprocal_sum_from_mask(new_mask)
                if new_sum > current_sum + 1e-12:
                    current_sum = new_sum
                    mask = new_mask
                    improved = True
        return mask, current_sum
    
    # Local search: 2-removal + fill
    def local_search_2swap(mask, max_iters=50):
        current_sum = reciprocal_sum_from_mask(mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1
            selected = [i for i in range(n) if mask & (1 << i)]
            for idx1 in range(len(selected)):
                if elapsed() > DEADLINE:
                    break
                i = selected[idx1]
                for idx2 in range(idx1 + 1, len(selected)):
                    j = selected[idx2]
                    mask_without = mask & ~(1 << i) & ~(1 << j)
                    new_mask = fill_mask(mask_without)
                    new_sum = reciprocal_sum_from_mask(new_mask)
                    if new_sum > current_sum + 1e-12:
                        current_sum = new_sum
                        mask = new_mask
                        improved = True
                        selected = [i for i in range(n) if mask & (1 << i)]
                        break
                if improved:
                    break
        return mask, current_sum
    
    # Local search: 3-removal + fill
    def local_search_3swap(mask, max_iters=20):
        current_sum = reciprocal_sum_from_mask(mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1
            selected = [i for i in range(n) if mask & (1 << i)]
            ls = len(selected)
            for idx1 in range(ls):
                if elapsed() > DEADLINE:
                    break
                i = selected[idx1]
                for idx2 in range(idx1 + 1, ls):
                    if elapsed() > DEADLINE:
                        break
                    j = selected[idx2]
                    for idx3 in range(idx2 + 1, ls):
                        if elapsed() > DEADLINE:
                            break
                        k = selected[idx3]
                        mask_without = mask & ~(1 << i) & ~(1 << j) & ~(1 << k)
                        new_mask = fill_mask(mask_without)
                        new_sum = reciprocal_sum_from_mask(new_mask)
                        if new_sum > current_sum + 1e-12:
                            current_sum = new_sum
                            mask = new_mask
                            improved = True
                            selected = [i for i in range(n) if mask & (1 << i)]
                            break
                    if improved:
                        break
                if improved:
                    break
        return mask, current_sum
    
    # Run local search on best
    selected_mask = best_mask
    selected_mask, best_sum = local_search_1swap(selected_mask)
    
    if elapsed() < DEADLINE - 10:
        selected_mask, local_sum = local_search_2swap(selected_mask)
        if local_sum > best_sum:
            best_sum = local_sum
            best_mask = selected_mask
    
    if elapsed() < DEADLINE - 5:
        selected_mask, local_sum = local_search_3swap(selected_mask, max_iters=10)
        if local_sum > best_sum:
            best_sum = local_sum
            best_mask = selected_mask
    
    # More random restarts with local search
    for _ in range(25):
        if elapsed() > DEADLINE - 8:
            break
        order_r = list(range(n))
        random.shuffle(order_r)
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            if random.random() < 0.8:
                if adjusted_weights[order_r[i]] < adjusted_weights[order_r[j]]:
                    order_r[i], order_r[j] = order_r[j], order_r[i]
        mask_r = fill_mask(greedy_construct(order_r))
        mask_r, sum_r = local_search_1swap(mask_r, max_iters=30)
        if sum_r > best_sum:
            best_sum = sum_r
            best_mask = mask_r
    
    # Final polish: 2-swap then 3-swap on best
    if elapsed() < DEADLINE - 3:
        best_mask, final_sum = local_search_2swap(best_mask, max_iters=20)
        if final_sum > best_sum:
            best_sum = final_sum
    
    if elapsed() < DEADLINE - 1:
        best_mask, final_sum = local_search_3swap(best_mask, max_iters=5)
        if final_sum > best_sum:
            best_sum = final_sum
    
    # Convert back to numbers
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    return result
# EVOLVE-BLOCK-END
