"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # For each pair (a,b), find all z such that {a,b,z} is a bad triple
    # i.e., lcm(a,b) == lcm(a,z) == lcm(b,z)
    # Store as: pair_witness[(a,b)] = set of z values
    pair_witness = defaultdict(set)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for z in range(1, n + 1):
                if z == a or z == b:
                    continue
                if lcm_table[a][z] == L_ab and lcm_table[b][z] == L_ab:
                    pair_witness[(a, b)].add(z)
    
    # For each element x, precompute which pairs (a,b) would be "blocked" by x
    # i.e., x is in pair_witness[(a,b)]
    # This is the same as the conflicts structure but from the pair's perspective
    # conflicts_blocked_by[x] = set of pairs (a,b) such that x is a witness for (a,b)
    blocked_by = [set() for _ in range(n + 1)]
    for (a, b), witnesses in pair_witness.items():
        for z in witnesses:
            blocked_by[z].add((a, b))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        # For each pair in current_set, check if x is a witness for that pair
        # Equivalently: check if any pair in current_set is in blocked_by[x]
        lst = sorted(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if (a, b) in blocked_by[x]:
                    return False
        return True
    
    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(initial_set, max_iters=200):
        """Local search: try to improve by removing one and adding another, or just adding."""
        current_set = set(initial_set)
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 2 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to add an element first (cheaper)
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            for out_elem in list(current_set):
                if time.time() > deadline - 2:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    random.seed(42)
    best_set = set()
    
    # Strategy: multiple restarts with different heuristics
    # 1. Random order
    # 2. Sorted by number of conflicts (least conflicted first)
    # 3. Sorted by value (ascending/descending)
    # 4. Random with bias toward numbers with fewer blocked pairs
    
    # Precompute conflict count for each element
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = len(blocked_by[x])
    
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Different ordering strategies
        if i % 4 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 4 == 1:
            order = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
        elif i % 4 == 2:
            order = list(range(1, n + 1))
        else:
            order = list(range(n, 0, -1))
        
        s = greedy_add(order)
        s_improved = local_search(s)
        if len(s_improved) > len(best_set):
            best_set = s_improved
    
    # Final pass: try to add any remaining elements
    if time.time() < deadline - 2:
        for in_elem in range(1, n + 1):
            if in_elem in best_set:
                continue
            if can_add(in_elem, best_set):
                best_set.add(in_elem)
    
    # Verify and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
