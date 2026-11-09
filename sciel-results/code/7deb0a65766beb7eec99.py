"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(max_n):
    """Precompute LCM table for all pairs in [1, max_n]."""
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _precompute_conflict_masks(max_n, lcm_table):
    """
    For each element x, compute a bitmask of all other elements y such that
    there exists a third element z where {x, y, z} forms a bad triple.
    Actually, we need: for each pair (a,b), the set of c such that lcm(a,b)==lcm(a,c)==lcm(b,c).
    
    Better approach: For each element x, compute mask_x = set of all y != x such that
    x and y are "conflicted" in the sense that adding both x and y to a set could lead to
    a bad triple with some z. But actually the constraint is on triples.
    
    Let's think differently. A triple {a,b,c} is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    For a given set S, it's valid if no such triple is fully contained in S.
    
    For greedy with bitmask: we can precompute for each pair (a,b) the list of c's that
    would form a bad triple. Then when we have selected set, we need to check if adding x
    creates a bad triple. That means: for any two already-selected elements a,b, check if
    x is in bad_triples[a][b].
    
    More efficiently: for each element x, precompute conflict_mask[x] = bitmask of elements
    y such that there exists some z where {x,y,z} is bad. But that's not quite right either,
    because the conflict depends on which other elements are already selected.
    
    Actually, the cleanest approach: precompute for each pair (a,b) a bitmask of all c such
    that (a,b,c) is a bad triple. Then when considering adding x to selected set, for each
    pair (a,b) in selected, check if x is in bad_pair_mask[a][b].
    
    But that's O(k^2) per insertion where k is current set size.
    
    Alternative: for each x, precompute a bitmask conflict_with[x] = union over all pairs
    (a,b) where x is the "third" element... no, that doesn't work directly.
    
    Let's just use the pair-based approach. Precompute bad_pair_mask[a][b] = bitmask of c's.
    """
    n = max_n
    # bad_pair_mask[a][b] = bitmask of all c > max(a,b) or c != a,b such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_pair_mask = [[0] * (n + 1) for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_pair_mask[a][b] = mask
            bad_pair_mask[b][a] = mask
    
    return bad_pair_mask

def _greedy_with_order(order, bad_pair_mask, max_n):
    """
    Greedily add elements in the given order, checking conflicts.
    Returns the selected set as a list.
    """
    selected = []
    selected_mask = 0  # bitmask of selected elements
    
    for x in order:
        # Check if adding x conflicts with any pair in selected
        conflict = False
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a = selected[i]
                b = selected[j]
                # Check if x is in bad_pair_mask[a][b]
                if bad_pair_mask[a][b] & (1 << x):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            selected.append(x)
            selected_mask |= (1 << x)
    
    return selected

def _local_search_improve(selected, bad_pair_mask, max_n, max_iterations=100):
    """
    Try to improve the selected set by removing and re-adding elements.
    Simple local search: try swapping or adding.
    """
    selected_set = set(selected)
    selected_list = list(selected)
    
    for iteration in range(max_iterations):
        improved = False
        # Try adding each unselected element
        for x in range(1, max_n + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            can_add = True
            for i in range(len(selected_list)):
                for j in range(i + 1, len(selected_list)):
                    a = selected_list[i]
                    b = selected_list[j]
                    if bad_pair_mask[a][b] & (1 << x):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                selected_list.append(x)
                selected_set.add(x)
                improved = True
                break
        if not improved:
            break
    
    return selected_list

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    max_n = 128
    lcm_table = _precompute_lcms(max_n)
    bad_pair_mask = _precompute_conflict_masks(max_n, lcm_table)
    
    # Generate multiple diverse orderings
    base_order = list(range(1, max_n + 1))
    
    # Ordering 1: natural order
    orders = [base_order]
    
    # Ordering 2: reverse
    orders.append(list(reversed(base_order)))
    
    # Ordering 3: random
    for _ in range(20):
        random_order = base_order.copy()
        random.shuffle(random_order)
        orders.append(random_order)
    
    # Ordering 4: sorted by number of prime factors (ascending - fewer primes first)
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    orders.append(sorted(base_order, key=num_prime_factors))
    orders.append(sorted(base_order, key=num_prime_factors, reverse=True))
    
    # Ordering 5: sorted by value of lcm with others (heuristic)
    def lcm_sum(n):
        return sum(lcm_table[n][j] for j in range(1, max_n + 1) if j != n)
    
    orders.append(sorted(base_order, key=lcm_sum))
    orders.append(sorted(base_order, key=lcm_sum, reverse=True))
    
    # Run greedy on each ordering and keep the best
    best_selected = []
    for order in orders:
        if time.time() > deadline:
            break
        selected = _greedy_with_order(order, bad_pair_mask, max_n)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    # Local search improvement
    if time.time() < deadline:
        best_selected = _local_search_improve(best_selected, bad_pair_mask, max_n, max_iterations=50)
    
    # More random restarts if time permits
    while time.time() < deadline - 2:
        random_order = base_order.copy()
        random.shuffle(random_order)
        selected = _greedy_with_order(random_order, bad_pair_mask, max_n)
        if len(selected) > len(best_selected):
            best_selected = selected
            best_selected = _local_search_improve(best_selected, bad_pair_mask, max_n, max_iterations=20)
    
    if not best_selected:
        best_selected = [1]
    
    return best_selected
# EVOLVE-BLOCK-END
