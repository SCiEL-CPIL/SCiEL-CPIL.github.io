"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # We want to find the minimum number of distinct divisors that sum to target.
    # This is equivalent to a shortest path problem where:
    # - States are possible sums (0 to target)
    # - Edges correspond to adding a divisor (if not already used)
    # 
    # However, tracking "used" divisors makes the state space huge (2^256).
    # Instead, we can use BFS on the number of terms.
    # 
    # Let's think differently: we want min k such that there exist k distinct divisors summing to target.
    # 
    # Since the number of divisors is 60 (let's verify: 5040 = 2^4 * 3^2 * 5 * 7, so (4+1)(2+1)(1+1)(1+1) = 5*3*2*2 = 60 divisors),
    # we can do a BFS where the state is (sum, frozenset of used divisors) but that's too expensive.
    # 
    # Alternative: Since we're minimizing the count, and the maximum possible sum with k divisors is bounded,
    # we can do BFS layer by layer (by number of divisors used).
    # 
    # But even better: use meet-in-the-middle or just iterate over possible counts.
    # 
    # Actually, let's reconsider. The greedy approach in the seed program likely gives a suboptimal count.
    # Let's estimate: 5040's largest divisors are 5040, 2520, 1680, 1260, 1008, 840, 720, 630, 560, 504, 420, 360, 336, 280, 252, 240, 210, 180, 168, 140, 126, 120, 105, 90, 84, 72, 70, 63, 56, 48, 42, 40, 36, 30, 28, 24, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.
    # 
    # Target is 4979. The largest divisor is 5040 which is > 4979, so we can't use it.
    # Next: 2520 + 1680 = 4200, remainder 779. 779 is not a divisor. 
    # Let's just do a proper search.
    # 
    # Given the small number of divisors (60), we can use BFS with state = (current_sum, tuple of used divisor indices).
    # But 2^60 is too large.
    # 
    # Better approach: Since we're minimizing the number of terms, let's try k=1, k=2, k=3, ... and check if any combination of k distinct divisors sums to target.
    # For small k, this is feasible.
    # 
    # k=1: Check if target is a divisor. 4979 is not a divisor of 5040 (5040/4979 is not integer).
    # k=2: Check if there exist two distinct divisors d1, d2 with d1+d2=4979.
    # k=3: Check if there exist three distinct divisors with sum 4979.
    # etc.
    # 
    # Let's implement this.
    
    divisors_set = set(divisors)
    divisors_list = sorted(divisors, reverse=True)
    
    # k=1
    if target in divisors_set:
        return [target]
    
    # k=2
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in divisors_set and d2 != d1:
            return [d1, d2]
    
    # k=3
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            d3 = target - d1 - d2
            if d3 in divisors_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]
    
    # k=4
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            for k in range(j + 1, len(divisors_list)):
                d3 = divisors_list[k]
                if d1 + d2 + d3 >= target:
                    continue
                d4 = target - d1 - d2 - d3
                if d4 in divisors_set and d4 != d1 and d4 != d2 and d4 != d3:
                    return [d1, d2, d3, d4]
    
    # For k>=5, we can use a more systematic approach.
    # Let's use BFS with memoization on (sum, frozenset of used indices) but prune heavily.
    # Actually, given the constraints, let's just do a DFS with pruning for increasing k.
    
    # Let's try a different approach: use the fact that the sum of all divisors of 5040 is known.
    # Sum of divisors of 5040 = (1+2+4+8+16)(1+3+9)(1+5)(1+7) = 31 * 13 * 6 * 8 = 31*13=403, 403*6=2418, 2418*8=19344.
    # So total sum is 19344, which is much larger than 4979.
    
    # Let's do a BFS where we track (sum, used_mask) but use a bitset for used divisors.
    # 60 divisors means 60-bit mask, which fits in a Python int.
    
    from functools import lru_cache
    
    # Map each divisor to an index
    divisor_to_idx = {d: i for i, d in enumerate(divisors_list)}
    num_divs = len(divisors_list)
    
    # BFS by number of terms
    # State: (sum, mask) -> minimum number of terms to reach this state
    # We want to reach (target, any_mask) with minimum terms.
    
    # Use a queue: (sum, mask, count)
    # But to avoid revisiting states, we can use a dict: (sum, mask) -> min_count
    
    # Actually, let's just do BFS level by level.
    
    # Level 0: sum=0, mask=0
    # Level 1: all single divisors
    # Level 2: all pairs, etc.
    
    # Since we already checked k=1,2,3,4 above, let's start from k=5.
    # But to be safe, let's restart the BFS from k=1.
    
    # BFS
    from collections import defaultdict
    
    # visited: dict from (sum, mask) to True
    # But mask can be large. Let's use a set of (sum, mask).
    
    # Actually, for each sum, we only need to know which masks have been visited.
    # But there are 2^60 possible masks, so we can't store all.
    
    # Alternative: For each sum, keep track of the set of masks that have been visited.
    # This is still potentially large.
    
    # Let's just do a DFS with pruning for k=5,6,... up to some limit.
    
    # Given the time constraint, let's assume k is small (<=10) and do a recursive search.
    
    def dfs(start_idx, remaining, used, current_list):
        if remaining == 0:
            return current_list
        if remaining < 0:
            return None
        if start_idx >= num_divs:
            return None
        
        # Pruning: if even taking all remaining divisors from start_idx can't reach remaining, return None
        # Precompute suffix sums
        # For simplicity, let's just try each divisor
        for i in range(start_idx, num_divs):
            d = divisors_list[i]
            if d > remaining:
                continue
            # Try including d
            result = dfs(i + 1, remaining - d, used | (1 << i), current_list + [d])
            if result is not None:
                return result
        return None
    
    # Try k=5,6,...
    for k in range(5, 11):
        # We need to find k distinct divisors summing to target
        # Use DFS with the constraint that we pick exactly k divisors
        # This is still exponential, but with pruning it might work for small k
        
        # Let's use a different pruning: sort divisors in descending order and use the fact that
        # the maximum possible sum with k divisors starting from index i is bounded.
        
        # Precompute suffix sums for the top k divisors
        # Actually, let's just do the DFS and hope it's fast enough for k<=10
        
        # To make it faster, let's limit the search space by only considering divisors <= target
        valid_divs = [(d, i) for i, d in enumerate(divisors_list) if d <= target]
        
        def dfs_k(idx, remaining, count, current_list):
            if count == 0:
                return current_list if remaining == 0 else None
            if idx >= len(valid_divs):
                return None
            # Pruning: if remaining < 0 or remaining > sum of largest 'count' divisors from idx, return None
            # For simplicity, just do the search
            for j in range(idx, len(valid_divs)):
                d, orig_idx = valid_divs[j]
                if d > remaining:
                    continue
                result = dfs_k(j + 1, remaining - d, count - 1, current_list + [d])
                if result is not None:
                    return result
            return None
        
        result = dfs_k(0, target, k, [])
        if result is not None:
            return result
    
    # If we get here, we haven't found a solution with k<=10.
    # Fall back to the greedy approach
    remainder = target
    chosen = []
    for d in divisors_list:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0
    return chosen

# EVOLVE-BLOCK-END
