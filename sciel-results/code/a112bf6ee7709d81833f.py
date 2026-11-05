# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n_val = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n_val + 1) if n_val % d == 0]
    divisors.sort()
    n_divs = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    total_sum = prefix[n_divs]
    
    # Helper to check if target can be formed by k distinct divisors from divisors[start_idx:]
    # Returns the list of divisors if found, else None
    def find_k_divisors(k, start_idx, remaining_sum):
        if k == 0:
            return [] if remaining_sum == 0 else None
        
        if start_idx >= n_divs:
            return None
            
        # Pruning: not enough elements left
        if n_divs - start_idx < k:
            return None
            
        # Pruning: min possible sum
        min_possible = prefix[start_idx + k] - prefix[start_idx]
        if remaining_sum < min_possible:
            return None
            
        # Pruning: max possible sum
        if start_idx <= n_divs - k:
            max_possible = total_sum - prefix[n_divs - k]
        else:
            max_possible = total_sum - prefix[start_idx]
            
        if remaining_sum > max_possible:
            return None
            
        # Try each divisor from start_idx
        for i in range(start_idx, n_divs):
            d = divisors[i]
            
            # If we pick d, the remaining sum must be achievable by k-1 divisors from i+1 onwards
            new_remaining = remaining_sum - d
            if new_remaining < 0:
                break # Since divisors are sorted, further d will be larger
            
            # Quick check: can we form new_remaining with k-1 divisors from i+1?
            # Min sum for k-1 from i+1
            if i + 1 + (k - 1) <= n_divs:
                min_next = prefix[i + 1 + (k - 1)] - prefix[i + 1]
                if new_remaining < min_next:
                    # If new_remaining is too small for the smallest k-1 divisors after i,
                    # it will be even smaller for larger d (larger i), so we can break.
                    break
            else:
                continue

            # Max sum for k-1 from i+1
            if i + 1 <= n_divs - (k - 1):
                max_next = total_sum - prefix[n_divs - (k - 1)]
            else:
                max_next = total_sum - prefix[i + 1]
                
            if new_remaining > max_next:
                pass

            # Recursive call
            res = find_k_divisors(k - 1, i + 1, new_remaining)
            if res is not None:
                return [d] + res
                
        return None
    
    # k=1
    if target in divisor_set:
        return [target]
        
    # k=2: O(N) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement > d:
            return [d, complement]
            
    # k=3: O(N^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            c = target - a - b
            if c in divisor_set and c > b:
                return [a, b, c]
                
    # k=4: Meet-in-the-middle
    pair_sums = {}
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
            
    for s1, pairs1 in pair_sums.items():
        s2 = target - s1
        if s2 not in pair_sums:
            continue
        if s2 < s1:
            continue
        pairs2 = pair_sums[s2]
        
        if s1 == s2:
            for p1_idx in range(len(pairs1)):
                i1, j1 = pairs1[p1_idx]
                for p2_idx in range(p1_idx + 1, len(pairs1)):
                    i2, j2 = pairs1[p2_idx]
                    if len({i1, j1, i2, j2}) == 4:
                        return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
        else:
            for i1, j1 in pairs1:
                for i2, j2 in pairs2:
                    if len({i1, j1, i2, j2}) == 4:
                        return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]

    # k >= 5: Backtracking with strong pruning
    for k in range(5, 257):
        if time.time() > deadline:
            break
            
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        if target < min_sum_k or target > max_sum_k:
            continue
            
        res = find_k_divisors(k, 0, target)
        if res is not None:
            return res
            
    # Fallback: Greedy
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder and d not in chosen:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
                
    return chosen
# EVOLVE-BLOCK-END
