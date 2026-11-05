"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 128
    
    # Pre-compute lcms for all pairs
    lcms = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[(i, j)] = lcm(i, j)
    
    # Find all bad triples and build conflict structures
    # conflict_map[v] = set of (u, w) such that {v, u, w} is a bad triple
    conflict_map = defaultdict(set)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[(a, b)]
            for c in range(b + 1, N + 1):
                lac = lcms[(a, c)]
                lbc = lcms[(b, c)]
                if lab == lac == lbc:
                    conflict_map[a].add((b, c))
                    conflict_map[b].add((a, c))
                    conflict_map[c].add((a, b))
    
    if time.time() > deadline:
        return list(range(64, 129))
    
    # Degree: number of conflicting pairs for each element
    degree = {v: len(conflict_map[v]) for v in range(1, N + 1)}
    
    # Helper: check if adding v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                return False
        return True
    
    # Helper: check if removing v_out and adding v_in keeps the set valid
    def can_swap_remove_add(v_out, v_in, selected_set):
        temp_set = selected_set - {v_out}
        for u, w in conflict_map[v_in]:
            if u in temp_set and w in temp_set:
                return False
        return True
    
    # Greedy selection
    def greedy_select(order):
        selected_set = set()
        for v in order:
            if time.time() > deadline:
                break
            if can_add(v, selected_set):
                selected_set.add(v)
        return selected_set
    
    # Try degree-based greedy
    order1 = sorted(range(1, N + 1), key=lambda x: degree[x])
    best_set = greedy_select(order1)
    
    # Randomized greedy with multiple restarts
    random.seed(42)
    
    if time.time() < deadline - 10:
        for restart in range(50):
            if time.time() > deadline - 2:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Bias: put low-degree elements earlier
            order.sort(key=lambda x: (degree[x] * 0.3 + random.random() * 0.7))
            s = greedy_select(order)
            if len(s) > len(best_set):
                best_set = s
    
    # Local search: try to add more elements
    def try_add_more(selected_set):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in range(1, N + 1):
                if v not in selected_set and can_add(v, selected_set):
                    selected_set.add(v)
                    improved = True
        return selected_set
    
    best_set = try_add_more(best_set)
    
    # Local search: 1-swap improvements to potentially enable more additions
    def local_search_1swap(selected_set):
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            selected_list = list(selected_set)
            unselected = [v for v in range(1, N + 1) if v not in selected_set]
            
            for v_out in selected_list:
                if time.time() > deadline - 1:
                    break
                for v_in in unselected:
                    if can_swap_remove_add(v_out, v_in, selected_set):
                        new_set = selected_set - {v_out} | {v_in}
                        # Try to add more after swap
                        new_set = try_add_more(new_set)
                        if len(new_set) > len(selected_set):
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
        return selected_set
    
    if time.time() < deadline - 5:
        best_set = local_search_1swap(best_set)
        best_set = try_add_more(best_set)
    
    # Final: ensure non-empty and return
    if not best_set:
        best_set = {1}
    
    return list(best_set)

# EVOLVE-BLOCK-END
