"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s limit
    
    # Precompute distinct prime factors for each number
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].append(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].append(x)
    
    # Group numbers by their prime factor set; for each group, keep only the max
    factor_groups = defaultdict(list)
    for i in range(1, n + 1):
        key = tuple(sorted(prime_factors[i]))
        factor_groups[key].append(i)
    
    representatives = {}
    for key, nums in factor_groups.items():
        representatives[key] = max(nums)
    
    # Build list of (weight, prime_set) for each representative
    rep_items = [(weight, primes) for primes, weight in representatives.items()]
    rep_items.sort(key=lambda x: -x[0])  # descending weight
    
    num_items = len(rep_items)
    
    # Precompute conflict between items (share at least one prime)
    item_conflict = [set() for _ in range(num_items)]
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if set(rep_items[i][1]) & set(rep_items[j][1]):
                item_conflict[i].add(j)
                item_conflict[j].add(i)
    
    best_sum = 0
    best_selection = []
    
    def solve_packing(idx, selected, current_sum, available):
        nonlocal best_sum, best_selection
        
        if time.time() > deadline:
            return
        
        if idx == num_items:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(selected)
            return
        
        # Upper bound
        ub = current_sum
        for j in range(idx, num_items):
            if j in available:
                ub += rep_items[j][0]
        if ub <= best_sum:
            return
        
        item_idx = idx
        weight, primes = rep_items[item_idx]
        
        # Option 1: Include if available and compatible with selected
        if item_idx in available:
            compatible = True
            for s in selected:
                if s in item_conflict[item_idx]:
                    compatible = False
                    break
            if compatible:
                new_selected = selected + [item_idx]
                new_available = available - item_conflict[item_idx] - {item_idx}
                solve_packing(item_idx + 1, new_selected, current_sum + weight, new_available)
        
        # Option 2: Exclude
        solve_packing(item_idx + 1, selected, current_sum, available)
    
    initial_available = set(range(num_items))
    solve_packing(0, [], 0, initial_available)
    
    if best_selection:
        result = [rep_items[i][0] for i in best_selection]
    else:
        result = [1]
    
    # Ensure Python ints and return as list
    result = [int(x) for x in result]
    return result

# EVOLVE-BLOCK-END
