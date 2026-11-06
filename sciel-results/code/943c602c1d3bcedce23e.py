import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    D = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums for pruning
    prefix = [0] * (D + 1)
    for i in range(D):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    def range_sum(lo, hi):
        """Sum of divisors[lo:hi]"""
        return prefix[hi] - prefix[lo]

    # Greedy fallback solution
    def greedy_solution():
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            return chosen
        return []

    # Specialized fast check for k=2
    def solve_k2(target_val):
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            d2 = target_val - d1
            if d2 in divisor_set and d2 != d1:
                return [d1, d2]
        return None

    # Specialized fast check for k=3
    def solve_k3(target_val):
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            rem1 = target_val - d1
            for j in range(i + 1, D):
                d2 = divisors[j]
                if d2 >= rem1:
                    continue
                rem2 = rem1 - d2
                if rem2 in divisor_set and rem2 < d2:
                    return [d1, d2, rem2]
        return None

    # Meet-in-the-middle for k >= 4
    def solve_k_mitm(k, target_val):
        if k > 8:
            return None  # Too large for MITM
        
        # Split divisors into two halves
        mid = D // 2
        left = divisors[:mid]
        right = divisors[mid:]
        
        # Split k into two parts
        k1 = k // 2
        k2 = k - k1
        
        # Generate all sums of k1 elements from left
        left_sums = {}
        for combo in combinations(left, k1):
            s = sum(combo)
            if s <= target_val:
                if s not in left_sums:
                    left_sums[s] = combo
        
        # Generate all sums of k2 elements from right
        right_sums = {}
        for combo in combinations(right, k2):
            s = sum(combo)
            if s <= target_val:
                if s not in right_sums:
                    right_sums[s] = combo
        
        # Check for pairs that sum to target_val
        for s1, combo1 in left_sums.items():
            s2 = target_val - s1
            if s2 in right_sums:
                combo2 = right_sums[s2]
                # Verify no overlap (shouldn't happen since left and right are disjoint)
                all_elements = list(combo1) + list(combo2)
                if len(set(all_elements)) == len(all_elements):
                    return all_elements
        
        return None

    # General backtracking for k >= 9 (fallback)
    def solve_k_general(k, target_val):
        if k == 0:
            return [] if target_val == 0 else None
        if k == 1:
            if target_val in divisor_set:
                return [target_val]
            return None
            
        def backtrack(start, k_rem, target_rem, current_path):
            if k_rem == 0:
                if target_rem == 0:
                    return current_path[:]
                return None
            
            if len(divisors) - start < k_rem:
                return None
                
            # Pruning: Max sum
            if start + k_rem > len(divisors):
                return None
            max_possible = range_sum(start, start + k_rem)
            if max_possible < target_rem:
                return None
                
            # Pruning: Min sum
            min_possible = range_sum(D - k_rem, D)
            if min_possible > target_rem:
                return None

            for i in range(start, len(divisors)):
                d = divisors[i]
                if d > target_rem:
                    continue
                
                remaining = target_rem - d
                if k_rem - 1 == 0:
                    if remaining == 0:
                        return current_path + [d]
                    continue
                
                if len(divisors) - (i + 1) < k_rem - 1:
                    break
                
                if i + 1 + (k_rem - 1) <= len(divisors):
                    max_next = range_sum(i + 1, i + 1 + k_rem - 1)
                    if d + max_next < target_rem:
                        continue
                else:
                    break
                    
                min_next = range_sum(D - (k_rem - 1), D)
                if d + min_next > target_rem:
                    continue

                current_path.append(d)
                result = backtrack(i + 1, k_rem - 1, remaining, current_path)
                if result:
                    return result
                current_path.pop()
            
            return None
        
        return backtrack(0, k, target_val, [])

    # Try k from 1 upwards
    for k in range(1, D + 1):
        if time.time() > deadline:
            return greedy_solution()
            
        result = None
        if k == 1:
            if target in divisor_set:
                result = [target]
        elif k == 2:
            result = solve_k2(target)
        elif k == 3:
            result = solve_k3(target)
        elif k <= 8:
            result = solve_k_mitm(k, target)
        else:
            result = solve_k_general(k, target)
            
        if result is not None:
            return result
            
    return greedy_solution()
