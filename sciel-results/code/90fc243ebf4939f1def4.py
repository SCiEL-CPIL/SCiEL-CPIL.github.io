"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Build conflict graph
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if lcm(candidates[i], candidates[j]) <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    weights = [1.0 / candidates[i] for i in range(n)]
    
    # Sort by weight descending
    order = sorted(range(n), key=lambda i: -weights[i])
    
    # Precompute neighbor masks
    neighbor_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        neighbor_mask[i] = mask
    
    full_mask = (1 << n) - 1
    
    best_weight = 0.0
    best_selection = [False] * n
    
    def compute_available_weight_sum(idx, available_mask):
        total = 0.0
        for j in range(idx, n):
            v = order[j]
            if (available_mask >> v) & 1:
                total += weights[v]
        return total
    
    def branch_and_bound(idx, current_weight, available_mask):
        nonlocal best_weight, best_selection, deadline
        
        if time.time() > deadline:
            return
        
        if idx == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_selection = [(available_mask >> order[i]) & 1 for i in range(n)]
            return
        
        avail_sum = compute_available_weight_sum(idx, available_mask)
        ub = current_weight + avail_sum
        if ub <= best_weight:
            return
        
        v = order[idx]
        
        # Try include first (higher weight vertices first)
        if (available_mask >> v) & 1:
            new_available = available_mask & ~(neighbor_mask[v] | (1 << v))
            branch_and_bound(idx + 1, current_weight + weights[v], new_available)
        
        new_available2 = available_mask & ~(1 << v)
        branch_and_bound(idx + 1, current_weight, new_available2)
    
    branch_and_bound(0, 0.0, full_mask)
    
    if not any(best_selection):
        return [2]
    
    result = [candidates[i] for i in range(n) if best_selection[i]]
    return result
# EVOLVE-BLOCK-END
