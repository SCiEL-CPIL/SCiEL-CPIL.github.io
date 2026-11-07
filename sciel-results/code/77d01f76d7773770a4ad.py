import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin before 60s limit
    
    n = 256
    
    # Precompute prime factors for each number 1..256
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # factors[i] = set of prime factors of i (for i in 1..256)
    # factors[1] = empty set
    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = prime_factors(i)
    
    # Check compatibility: two numbers are compatible if their prime factor sets are disjoint
    def compatible(a, b):
        if a == b:
            return True
        return len(factors[a] & factors[b]) == 0
    
    # Build incompatibility graph: vertex i is connected to vertex j if NOT coprime
    # For branch-and-bound MWIS, we work with the graph where edges = incompatible pairs
    # An independent set in this graph = pairwise coprime set
    
    # Represent vertices as 0-indexed (vertex i represents number i+1)
    # incompat[i] = bitmask of vertices incompatible with vertex i
    # Use Python integers as bitmasks for efficiency
    incompat = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not compatible(i + 1, j + 1):
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # Greedy solution to get initial bound
    def greedy_solution():
        selected = []
        remaining = set(range(n))
        
        # Sort by weight descending
        for v in sorted(remaining, key=lambda x: x + 1, reverse=True):
            if v not in remaining:
                continue
            # Check if v is compatible with all selected
            can_add = True
            for s in selected:
                if incompat[v] & (1 << s):
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                remaining.discard(v)
        
        return selected
    
    best_selection = greedy_solution()
    best_weight = sum(v + 1 for v in best_selection)
    
    # Branch and bound for Maximum Weight Independent Set
    def solve_mwis(vertices_mask, current_selection, current_weight):
        nonlocal best_weight, best_selection
        
        if time.time() > deadline:
            return
        
        if vertices_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_selection = list(current_selection)
            return
        
        # Extract vertex list from mask
        remaining = []
        m = vertices_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            remaining.append(v)
            m ^= lsb
        
        if not remaining:
            if current_weight > best_weight:
                best_weight = current_weight
                best_selection = list(current_selection)
            return
        
        # Quick upper bound: current_weight + sum of all remaining weights
        sum_remaining = sum(i + 1 for i in remaining)
        if current_weight + sum_remaining <= best_weight:
            return
        
        # Pick a vertex to branch on: choose the one with highest weight
        v = max(remaining, key=lambda x: x + 1)
        
        # Branch 1: include v
        # Remove v and all its neighbors from remaining
        new_mask = vertices_mask & ~(1 << v)
        # Remove neighbors of v
        neigh_mask = incompat[v] & vertices_mask
        new_mask &= ~neigh_mask
        
        current_selection.append(v)
        solve_mwis(new_mask, current_selection, current_weight + v + 1)
        current_selection.pop()
        
        # Branch 2: exclude v
        new_mask = vertices_mask & ~(1 << v)
        solve_mwis(new_mask, current_selection, current_weight)
    
    # Start with all vertices
    initial_mask = (1 << n) - 1
    solve_mwis(initial_mask, [], 0)
    
    # Convert best selection (0-indexed) to 1-indexed
    result = [int(v + 1) for v in best_selection]
    
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result
