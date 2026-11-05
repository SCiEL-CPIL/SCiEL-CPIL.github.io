import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Precompute LCMs
    max_val = 128
    lcm_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # 2. Identify all conflicting triples
    # A triple (a, b, c) with a < b < c is conflicting if
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    conflicting_triples = []
    # Also build a map: for each pair (a,b), list of c's that form a conflict
    # This will speed up the "can I add x?" check
    # conflict_map[(a,b)] = set of c's such that {a,b,c} is conflicting
    conflict_map = {}
    
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            l_ab = lcm_table[a][b]
            for c in range(b + 1, max_val + 1):
                l_ac = lcm_table[a][c]
                if l_ac != l_ab:
                    continue
                l_bc = lcm_table[b][c]
                if l_bc == l_ab:
                    conflicting_triples.append((a, b, c))
                    if (a, b) not in conflict_map:
                        conflict_map[(a, b)] = set()
                    conflict_map[(a, b)].add(c)
                    if (a, c) not in conflict_map:
                        conflict_map[(a, c)] = set()
                    conflict_map[(a, c)].add(b)
                    if (b, c) not in conflict_map:
                        conflict_map[(b, c)] = set()
                        conflict_map[(b, c)].add(a)

    # 3. Calculate conflict count for each number
    conflict_count = [0] * (max_val + 1)
    for a, b, c in conflicting_triples:
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1

    # 4. Greedy construction
    # Sort numbers by conflict count (ascending)
    # Numbers with fewer conflicts are "safer" to include
    order = list(range(1, max_val + 1))
    order.sort(key=lambda x: conflict_count[x])
    
    def check_add(x, current_set):
        # Check if adding x to current_set creates a conflict
        # A conflict exists if there are a, b in current_set such that (a,b,x) is a conflicting triple
        # This is equivalent to: for any a in current_set, if (a,x) is a key in conflict_map,
        # then we must check if any b in conflict_map[(a,x)] is also in current_set.
        for a in current_set:
            if a == x: continue
            # Ensure a < x or just use sorted tuple for lookup?
            # conflict_map keys are (min, max) ? 
            # In step 2, I added (a,b) with a<b.
            # So I need to look up (min(a,x), max(a,x))
            key = (a, x) if a < x else (x, a)
            if key in conflict_map:
                # There are some c's that form a conflict with a and x
                # Check if any of those c's are in current_set
                for c in conflict_map[key]:
                    if c in current_set:
                        return False
        return True

    def greedy_construct(order):
        current_set = set()
        for x in order:
            if check_add(x, current_set):
                current_set.add(x)
        return current_set

    best_set = greedy_construct(order)
    best_size = len(best_set)

    # 5. Local Search / Simulated Annealing to improve
    # We can try to swap elements to increase size or maintain size with better structure
    # But greedy is likely very good here.
    # Let's try a few random permutations of the order to see if we get a larger set.
    
    # Optimization: check_add is O(|S| * avg_conflict_degree).
    # This is fast enough for multiple runs.
    
    for _ in range(20):
        if time.time() > deadline - 5: break
        random.shuffle(order)
        # Maybe mix sorted and random?
        # Or just random.
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 6. Local Search: Try to add elements not in set, or swap
    current_set = set(best_set)
    not_in_set = set(range(1, max_val + 1)) - current_set
    
    # Try to add any element from not_in_set
    # We can iterate through not_in_set and try to add if possible
    # If we can add, we do it.
    # This is a greedy improvement.
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in list(not_in_set):
            if check_add(x, current_set):
                current_set.add(x)
                not_in_set.remove(x)
                improved = True
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # Try to swap: remove x from set, add y from not_in_set
    # This might allow adding more elements later?
    # Actually, if we can add y without removing x, we would have done it in the loop above.
    # So swaps are only useful if y conflicts with some x, but removing x allows adding y AND some other z?
    # This is complex. The simple "add if possible" loop is a good first pass.
    
    # Let's do a few random restarts of the "add if possible" phase
    for _ in range(5):
        if time.time() > deadline - 1: break
        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set
        # Randomize the order of trying to add
        for x in list(not_in_set):
            if check_add(x, current_set):
                current_set.add(x)
                not_in_set.remove(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    return list(best_set)
