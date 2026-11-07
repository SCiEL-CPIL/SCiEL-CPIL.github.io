"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    nums = list(range(1, N + 1))
    n = len(nums)
    
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(n)]
    for i in range(n):
        x = nums[i]
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)
    
    # Two numbers are compatible if they share no prime factors
    # Build adjacency (conflict) as bitmask
    conflict = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if prime_factors[i] & prime_factors[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Sort by value descending
    order = sorted(range(n), key=lambda i: -nums[i])
    # Map: position in sorted order -> original index
    orig_at_pos = [order[p] for p in range(n)]
    
    # Build conflict in sorted order
    sorted_conflict = [0] * n
    for p in range(n):
        orig = orig_at_pos[p]
        mask = conflict[orig]
        sm = 0
        for q in range(n):
            if q != p:
                orig_q = orig_at_pos[q]
                if mask & (1 << orig_q):
                    sm |= (1 << q)
        sorted_conflict[p] = sm
    
    # Suffix sum for upper bound
    suffix_sum = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        suffix_sum[p] = suffix_sum[p + 1] + nums[orig_at_pos[p]]
    
    best_sum = 0
    best_positions = []
    
    def branch_and_bound(pos, current_sum, selected_mask):
        nonlocal best_sum, best_positions
        
        if time.time() > deadline:
            return
        
        if pos == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_positions = [p for p in range(n) if selected_mask & (1 << p)]
            return
        
        # Upper bound: current_sum + sum of remaining that could potentially be added
        # Quick bound: current_sum + suffix_sum[pos]
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        # Try including pos first (higher value first, better for finding good solutions early)
        if not (selected_mask & sorted_conflict[pos]):
            branch_and_bound(pos + 1, current_sum + nums[orig_at_pos[pos]], selected_mask | (1 << pos))
        
        if time.time() > deadline:
            return
        
        # Try excluding pos
        branch_and_bound(pos + 1, current_sum, selected_mask)
    
    branch_and_bound(0, 0, 0)
    
    result = [nums[orig_at_pos[p]] for p in best_positions]
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
