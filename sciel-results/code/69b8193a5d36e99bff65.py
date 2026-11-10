"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s
    
    N = 256
    # Candidates: 2..256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute conflict graph: edge (i,j) exists if lcm(candidates[i], candidates[j]) <= 256
    # We want an independent set in this graph (no conflicts)
    # Weight of vertex i = 1/candidates[i]
    
    # Build adjacency as bitmasks for fast operations
    # n can be up to 255, so we need big integers or arrays
    # Let's use Python's arbitrary precision integers as bitmasks
    
    # Precompute conflicts
    # conflicts[i] is a bitmask of vertices that conflict with i
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(candidates[i], candidates[j]) <= N:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Weights as Fractions for exact arithmetic
    weights = [Fraction(1, candidates[i]) for i in range(n)]
    
    # Sort candidates by weight descending (small numbers first) for better pruning
    # But we need to maintain index consistency... let's work with the original indices
    # and use a branch-and-bound that processes vertices in order of decreasing weight
    
    # Order vertices by weight descending
    order = sorted(range(n), key=lambda i: -weights[i])
    
    # For the B&B, we'll work with the original vertex indices but process in 'order'
    
    best_set = set()
    best_weight = Fraction(0, 1)
    
    # Greedy initial solution: take vertices in order of decreasing weight if compatible
    greedy_set = set()
    for i in order:
        if all(i not in conflicts[j] for j in greedy_set):
            greedy_set.add(i)
    greedy_weight = sum(weights[i] for i in greedy_set)
    best_set = set(greedy_set)
    best_weight = greedy_weight
    
    # Branch and bound
    # State: current set of selected vertices, current weight, next vertex to consider
    # We process vertices in 'order'
    
    # Upper bound: sum of remaining weights (fractional relaxation)
    # For tighter bound, we can use the fact that if we select a vertex, we can't select its neighbors
    
    def branch_and_bound(selected, weight, start_idx):
        nonlocal best_set, best_weight, deadline
        
        if time.time() > deadline:
            return
        
        if start_idx >= len(order):
            if weight > best_weight:
                best_weight = weight
                best_set = set(selected)
            return
        
        # Upper bound: current weight + sum of all remaining weights from start_idx
        # This is loose but fast. For a tighter bound, we could compute the max independent set
        # of the remaining graph, but that's expensive.
        
        # Quick upper bound
        remaining_sum = sum(weights[order[k]] for k in range(start_idx, len(order)))
        if weight + remaining_sum <= best_weight:
            return
        
        # Try including order[start_idx]
        v = order[start_idx]
        new_selected = selected | {v}
        # Check if v conflicts with any in selected
        conflict = False
        for u in selected:
            if (conflicts[v] >> u) & 1:
                conflict = True
                break
        
        if not conflict:
            branch_and_bound(new_selected, weight + weights[v], start_idx + 1)
        
        # Try excluding order[start_idx]
        branch_and_bound(selected, weight, start_idx + 1)
    
    # The simple B&B above might be too slow for n=255. Let's use a better approach.
    # Let's use a recursive B&B with a better upper bound.
    
    # Actually, let's try a different approach: local search with 2-swaps
    # Start from greedy, then try to improve
    
    current_set = set(greedy_set)
    current_weight = greedy_weight
    
    def get_conflicts_with_set(v, s):
        """Check if v conflicts with any element in set s"""
        for u in s:
            if (conflicts[v] >> u) & 1:
                return True
        return False
    
    # Local search: try single additions, single removals, 1-swaps, 2-swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a vertex
        for v in range(n):
            if v not in current_set and not get_conflicts_with_set(v, current_set):
                new_set = current_set | {v}
                new_weight = current_weight + weights[v]
                if new_weight > current_weight:
                    current_set = new_set
                    current_weight = new_weight
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try removing a vertex to allow adding two better ones
        # 1-swap: remove one, add one
        for v_remove in list(current_set):
            for v_add in range(n):
                if v_add in current_set or v_add == v_remove:
                    continue
                # Check if v_add conflicts with current_set - {v_remove}
                temp_set = current_set - {v_remove}
                if not get_conflicts_with_set(v_add, temp_set):
                    new_weight = current_weight - weights[v_remove] + weights[v_add]
                    if new_weight > current_weight:
                        current_set = temp_set | {v_add}
                        current_weight = new_weight
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # 2-swap: remove two, add two
        # This is O(k^2 * n^2) which could be expensive
        # Only do this if time allows
        if time.time() < deadline - 10:
            current_list = list(current_set)
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    v1, v2 = current_list[i], current_list[j]
                    temp_set = current_set - {v1, v2}
                    # Find best pair to add
                    best_pair_weight = Fraction(0, 1)
                    best_pair = None
                    for a in range(n):
                        if a in temp_set:
                            continue
                        if get_conflicts_with_set(a, temp_set):
                            continue
                        for b in range(a + 1, n):
                            if b in temp_set:
                                continue
                            if get_conflicts_with_set(b, temp_set):
                                continue
                            if get_conflicts_with_set(a, temp_set | {b}):
                                continue
                            pair_weight = weights[a] + weights[b]
                            if pair_weight > best_pair_weight:
                                best_pair_weight = pair_weight
                                best_pair = (a, b)
                    
                    if best_pair:
                        a, b = best_pair
                        new_weight = current_weight - weights[v1] - weights[v2] + best_pair_weight
                        if new_weight > current_weight:
                            current_set = temp_set | {a, b}
                            current_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            continue
    
    if current_weight > best_weight:
        best_weight = current_weight
        best_set = set(current_set)
    
    # Convert back to original values
    result = sorted([candidates[i] for i in best_set])
    return result
# EVOLVE-BLOCK-END
