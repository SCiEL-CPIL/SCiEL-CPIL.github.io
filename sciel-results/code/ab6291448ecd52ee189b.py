import time
import random
from math import lcm, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Pre-compute lcms for all pairs
    lcms = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcms[(i, j)] = lcm(i, j)
    
    # Find all bad triples and build conflict structures
    # conflict_with[v][u] = set of w such that {v,u,w} is a bad triple
    conflict_with = defaultdict(lambda: defaultdict(set))
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[(a, b)]
            for c in range(b + 1, N + 1):
                lac = lcms[(a, c)]
                lbc = lcms[(b, c)]
                if lab == lac == lbc:
                    conflict_with[a][b].add(c)
                    conflict_with[a][c].add(b)
                    conflict_with[b][a].add(c)
                    conflict_with[b][c].add(a)
                    conflict_with[c][a].add(b)
                    conflict_with[c][b].add(a)
    
    if time.time() > deadline:
        return list(range(64, 129))
    
    # Helper: check if adding v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u in conflict_with[v]:
            if u in selected_set:
                for w in conflict_with[v][u]:
                    if w in selected_set:
                        return False
        return True
    
    # Helper: check if removing v_out and adding v_in keeps the set valid
    def can_swap_remove_add(v_out, v_in, selected_set):
        temp_set = selected_set - {v_out}
        for u in conflict_with[v_in]:
            if u in temp_set:
                for w in conflict_with[v_in][u]:
                    if w in temp_set:
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
    
    # Strategy: Start with prime powers core, then greedy add
    def is_prime_power(n):
        if n == 1:
            return True
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                return False
        return True
    
    prime_powers = set()
    for x in range(1, N + 1):
        if is_prime_power(x):
            prime_powers.add(x)
    
    # Degree key for ordering
    def degree_key(x):
        return len(conflict_with[x])
    
    best_set = set()
    
    # Strategy 1: Start with prime powers, then greedy add remaining by degree
    core = set(prime_powers)
    remaining = [x for x in range(1, N + 1) if x not in core]
    remaining.sort(key=degree_key)
    s1 = set(core)
    for v in remaining:
        if time.time() > deadline:
            break
        if can_add(v, s1):
            s1.add(v)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Pure degree-based greedy (no core)
    order1 = sorted(range(1, N + 1), key=degree_key)
    s2 = greedy_select(order1)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: Randomized greedy with multiple restarts
    random.seed(42)
    
    if time.time() < deadline - 15:
        for restart in range(100):
            if time.time() > deadline - 3:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Bias: put low-degree elements earlier with some randomness
            order.sort(key=lambda x: (degree_key(x) * 0.2 + random.random() * 0.8))
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
            if time.time() > deadline - 2:
                break
            improved = False
            selected_list = list(selected_set)
            unselected = [v for v in range(1, N + 1) if v not in selected_set]
            
            for v_out in selected_list:
                if time.time() > deadline - 2:
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
