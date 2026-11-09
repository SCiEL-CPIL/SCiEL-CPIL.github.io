"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 80.0  # 80s guard, leaving 10s margin for 90s limit

    # 1. Generate divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to prioritize larger divisors (helps minimize count)
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # 2. Precompute suffix sums for max possible sum pruning
    # max_sum[i][k] = sum of the largest k divisors starting from index i
    # Since divisors are sorted descending, this is just sum(divisors[i : i+k])
    # We can compute this on the fly or precompute a 2D array.
    # Given num_divs=60, a 2D array is cheap.
    max_sum = [[0] * (num_divs + 1) for _ in range(num_divs + 1)]
    for i in range(num_divs):
        s = 0
        for k in range(1, num_divs - i + 1):
            s += divisors[i + k - 1]
            max_sum[i][k] = s

    # 3. Search for minimum k
    # Lower bound for k:
    # Largest divisors: 2520, 1680, 1260, 1008, 840...
    # 2520 + 1680 = 4200 < 4979
    # 2520 + 1680 + 1260 = 5460 > 4979
    # So k >= 3.
    
    for k in range(1, num_divs + 1):
        # Quick check: is max possible sum with k divisors >= target?
        if max_sum[0][k] < target:
            continue
        
        # Quick check: is min possible sum with k divisors <= target?
        # Min sum is sum of smallest k divisors.
        # Smallest divisors are at the end of the list.
        min_sum_k = sum(divisors[-k:])
        if min_sum_k > target:
            continue
            
        # DFS to find if a subset of size k sums to target
        found = False
        chosen = []
        
        def dfs(idx, count_left, current_sum):
            nonlocal found, chosen
            if found:
                return
            if time.time() > deadline:
                return # Timeout safety, though unlikely to be hit for small k
                
            if count_left == 0:
                if current_sum == target:
                    found = True
                return
                # Note: if found, we stop.
                
            if current_sum > target:
                return
            
            # Pruning: max possible sum from remaining choices
            # We need to pick 'count_left' items from indices idx..end
            # The maximum sum we can add is sum of the largest 'count_left' items in that range
            # Since sorted desc, these are divisors[idx] ... divisors[idx+count_left-1]
            if idx + count_left > num_divs:
                return
            if current_sum + max_sum[idx][count_left] < target:
                return
                
            # Iterate choices
            # We can skip choices that are too large?
            # If divisors[i] + current_sum > target, skip.
            # Also, we must leave enough items.
            # i can go from idx to num_divs - count_left
            
            for i in range(idx, num_divs - count_left + 1):
                if divisors[i] + current_sum > target:
                    continue
                
                chosen.append(divisors[i])
                dfs(i + 1, count_left - 1, current_sum + divisors[i])
                chosen.pop()
                if found:
                    return
                if time.time() > deadline:
                    return

        dfs(0, k, 0)
        
        if found:
            return chosen

    # Fallback: should not happen for this instance
    return []

# EVOLVE-BLOCK-END
