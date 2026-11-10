# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_fast(set_vals):
    """Fast check: no element is the mean of any subset of >=2 other elements."""
    n = len(set_vals)
    if n < 3:
        return True
    arr = np.array(set_vals, dtype=np.int64)
    max_val = int(arr.max())
    
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)  # m = n-1
        if m < 2:
            continue
        
        max_target = m * a
        # Always mask to keep bitsets manageable
        mask = (1 << (max_target + 1)) - 1
        
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 set
        
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                dp[k] &= mask
        
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    
    return True

def _check_addition_fast(current_set, x):
    """Check if adding x to current_set creates any violation.
    Only new violations can involve x as the mean, or existing elements becoming means
    of subsets that include x.
    """
    new_set = current_set + [x]
    n = len(new_set)
    if n < 3:
        return True
    
    # Check 1: Is x the mean of any subset of >=2 elements from current_set?
    if len(current_set) >= 2:
        others = current_set
        m = len(others)
        max_target = m * x
        if max_target <= 25000:
            mask = (1 << (max_target + 1)) - 1
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                    dp[k] &= mask
            for k in range(2, m + 1):
                target = k * x
                if target <= max_target and (dp[k] >> target) & 1:
                    return False
        else:
            # For large targets, use a different approach: check smaller k first
            # For k=2: need two elements a,b in current_set with a+b=2x
            # For k=3: need three elements with sum=3x, etc.
            # Just do the DP but with a tighter bound
            # Actually if max_target > 25000, the bitset is too large.
            # Use a set-based approach for small k
            for k in range(2, min(m+1, 6)):
                target = k * x
                # Check if there exist k distinct elements from current_set summing to target
                # Use recursive check for small k
                def _exists_subset_sum(elements, k, target):
                    if k == 0:
                        return target == 0
                    if k > len(elements):
                        return False
                    if target < 0:
                        return False
                    # Pruning: min possible sum of k smallest, max possible sum of k largest
                    if k > 0:
                        sorted_e = sorted(elements)
                        min_sum = sum(sorted_e[:k])
                        max_sum = sum(sorted_e[-k:])
                        if target < min_sum or target > max_sum:
                            return False
                    for i in range(len(elements)):
                        if _exists_subset_sum(elements[:i] + elements[i+1:], k-1, target - elements[i]):
                            return True
                    return False
                if _exists_subset_sum(others, k, target):
                    return False
    
    # Check 2: For each a in current_set, is a the mean of some subset of new_set that includes x?
    # Subset includes x and some elements from current_set \ {a}
    # Let S = {x} ∪ T where T ⊆ current_set \ {a}
    # sum(S) / |S| = a => x + sum(T) = a * (1 + |T|)
    # => sum(T) = a * (1 + |T|) - x
    # For |T| = t >= 1 (so |S| = t+1 >= 2): sum(T) = a*(t+1) - x
    # T is a subset of current_set \ {a} of size t
    
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m = len(others_no_a)
        if m < 1:
            continue
        
        max_t = m
        max_target = a * (max_t + 1) - x
        if max_target < 0:
            continue
        if max_target <= 25000:
            mask = (1 << (max_target + 1)) - 1
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others_no_a:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << elem)
                    dp[k] &= mask
            for t in range(1, m + 1):
                target = a * (t + 1) - x
                if 0 <= target <= max_target:
                    if (dp[t] >> target) & 1:
                        return False
        else:
            # Check small t values with recursive approach
            for t in range(1, min(m+1, 5)):
                target = a * (t + 1) - x
                if target < 0:
                    continue
                def _exists_subset_sum2(elements, k, target):
                    if k == 0:
                        return target == 0
                    if k > len(elements):
                        return False
                    if target < 0:
                        return False
                    if k > 0:
                        sorted_e = sorted(elements)
                        min_sum = sum(sorted_e[:k])
                        max_sum = sum(sorted_e[-k:])
                        if target < min_sum or target > max_sum:
                            return False
                    for i in range(len(elements)):
                        if _exists_subset_sum2(elements[:i] + elements[i+1:], k-1, target - elements[i]):
                            return True
                    return False
                if _exists_subset_sum2(others_no_a, t, target):
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    selected = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition_fast(selected, x):
            selected.append(x)
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Strategy 4: Try to add more elements to best_set
    if time.time() < deadline - 5:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 5: Local search - swap elements to free up space for additions
    if time.time() < deadline - 10 and len(best_set) < 48:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try replacing current[i] with some x not in current
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i+1:]
                    if _is_valid_fast(candidate):
                        # Check if we can add more elements
                        for y in range(1, 513):
                            if y in candidate:
                                continue
                            if len(candidate) >= 48:
                                break
                            if _check_addition_fast(candidate, y):
                                candidate.append(y)
                                improved = True
                                break
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break
    
    # Final: try to fill to 48
    if len(best_set) < 48 and time.time() < deadline - 2:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in current:
                continue
            if len(current) >= 48:
                break
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]
    
    return best_set[:48]
# EVOLVE-BLOCK-END
