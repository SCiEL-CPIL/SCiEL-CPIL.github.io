"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def lcm_lookup(a, b):
        return lcm_table[a][b]

    # Precompute bad triples: for each x, list of (a,b) pairs that form a bad triple with x
    bad_triples = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                lab = lcm_lookup(a, b)
                lax = lcm_lookup(a, x)
                lbx = lcm_lookup(b, x)
                if lab == lax == lbx:
                    bad_triples[x].append((a, b))
    
    def is_conflict_fast(x, selected_set):
        """Check if x conflicts with any pair in selected_set."""
        for a, b in bad_triples[x]:
            if a in selected_set and b in selected_set:
                return True
        return False
    
    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    def largest_prime_factor(x):
        max_p = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                max_p = d
                x //= d
            d += 1
        if x > 1:
            max_p = x
        return max_p
    
    best_result = None
    best_size = 0
    
    random.seed(42)
    
    orderings = []
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -1)))
    orderings.append(sorted(range(1, n + 1), key=omega))
    orderings.append(sorted(range(1, n + 1), key=omega, reverse=True))
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor))
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor, reverse=True))
    for p in [2, 3, 5, 7]:
        orderings.append(sorted(range(1, n + 1), key=lambda x: x % p))
    for _ in range(50):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)
    
    def greedy_construct(ordering):
        selected = []
        selected_set = set()
        for x in ordering:
            if not is_conflict_fast(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected, selected_set
    
    def local_search_add(selected, selected_set):
        """Try to add any unselected element. Returns (selected, selected_set)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                if not is_conflict_fast(x, selected_set):
                    selected.append(x)
                    selected_set.add(x)
                    improved = True
                    break
        return selected, selected_set
    
    def local_search_swap(selected, selected_set):
        """Try 1-swap: remove one element, try to add a better one."""
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                to_remove = selected[i]
                # Remove temporarily
                new_selected = [x for j, x in enumerate(selected) if j != i]
                new_set = set(new_selected)
                # Try to add any element
                for x in range(1, n + 1):
                    if x in new_set:
                        continue
                    if not is_conflict_fast(x, new_set):
                        if len(new_selected) + 1 > len(selected):
                            new_selected.append(x)
                            new_set.add(x)
                            selected = new_selected
                            selected_set = new_set
                            improved = True
                        break
                if improved:
                    break
        return selected, selected_set
    
    # Run greedy with all orderings
    for ordering in orderings:
        if time.time() > deadline - 2.0:
            break
        
        selected, selected_set = greedy_construct(ordering)
        selected, selected_set = local_search_add(selected, selected_set)
        
        if time.time() < deadline - 5.0:
            selected, selected_set = local_search_swap(selected, selected_set)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break
    
    # More randomized restarts
    while time.time() < deadline - 10.0:
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        selected, selected_set = greedy_construct(perm)
        selected, selected_set = local_search_add(selected, selected_set)
        if time.time() < deadline - 5.0:
            selected, selected_set = local_search_swap(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break
    
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
