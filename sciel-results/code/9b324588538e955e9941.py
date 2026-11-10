import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    num_divs = len(divisors)
    
    # Sort descending for better pruning
    divisors_desc = sorted(divisors, reverse=True)
    
    # Precompute cumulative sums for O(1) range sum queries
    cumsum = [0] * (num_divs + 1)
    for i in range(num_divs):
        cumsum[i+1] = cumsum[i] + divisors_desc[i]
    
    # For k=1: check if target is a divisor
    if target in divisors:
        return [target]
    
    # For k=2: check pairs
    if num_divs >= 2:
        divisor_set = set(divisors)
        for d in divisors:
            complement = target - d
            if complement in divisor_set and complement > d:
                return [d, complement]
    
    # For k=3: check triples
    if num_divs >= 3 and time.time() < deadline:
        divisor_set = set(divisors)
        for i in range(num_divs):
            d1 = divisors[i]
            if d1 >= target:
                break
            remaining = target - d1
            for j in range(i+1, num_divs):
                d2 = divisors[j]
                if d1 + d2 >= target:
                    break
                complement = remaining - d2
                if complement in divisor_set and complement > d2:
                    return [d1, d2, complement]
    
    def solve_with_k(k):
        """Find k distinct divisors summing to target using DFS with pruning."""
        if time.time() > deadline:
            return None
        
        if k > num_divs:
            return None
        max_possible = cumsum[k]
        if max_possible < target:
            return None
        min_possible = cumsum[num_divs] - cumsum[num_divs - k]
        if min_possible > target:
            return None
        
        path = []
        
        def dfs(idx, remaining_sum, slots_left):
            if time.time() > deadline:
                return False
            if slots_left == 0:
                return remaining_sum == 0
            if idx >= num_divs:
                return False
            
            if num_divs - idx < slots_left:
                return False
            
            if idx + slots_left > num_divs:
                return False
            max_sum = cumsum[idx + slots_left] - cumsum[idx]
            if remaining_sum > max_sum:
                return False
            
            min_sum = cumsum[num_divs] - cumsum[num_divs - slots_left]
            if remaining_sum < min_sum:
                return False
            
            if divisors_desc[idx] <= remaining_sum:
                path.append(divisors_desc[idx])
                if dfs(idx + 1, remaining_sum - divisors_desc[idx], slots_left - 1):
                    return True
                path.pop()
            
            if num_divs - idx - 1 >= slots_left:
                if dfs(idx + 1, remaining_sum, slots_left):
                    return True
            
            return False
        
        if dfs(0, target, k):
            return path[:]
        return None
    
    for k in range(4, min(num_divs + 1, 257)):
        if time.time() > deadline:
            break
        result = solve_with_k(k)
        if result is not None:
            return result
    
    # Fallback greedy
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
