"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    N = 128
    deadline = time.time() + 75  # safety margin before 90s kill

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            gcd_table[a][b] = gcd(a, b)
            gcd_table[b][a] = gcd_table[a][b]

    # Precompute bad triples: for each pair (a,b), store set of c such that {a,b,c} is bad
    # bad_with_pair[a][b] = set of c (c > b to avoid duplicates, but we need all)
    # Actually: for each pair (a,b), which c make {a,b,c} bad?
    # gcd(a,b) == gcd(b,c) == gcd(c,a)
    # So c must satisfy: gcd(b,c) == gcd(a,b) AND gcd(c,a) == gcd(a,b)
    
    # For each pair (a,b), precompute the set of valid c's that would form a bad triple
    # bad_c_for_pair[(a,b)] = set of c where {a,b,c} is bad
    bad_c_for_pair = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd_table[a][b]
            valid_cs = set()
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[b][c] == g and gcd_table[c][a] == g:
                    valid_cs.add(c)
            if valid_cs:
                bad_c_for_pair[(a, b)] = valid_cs
                bad_c_for_pair[(b, a)] = valid_cs

    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple."""
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                key = (a, b)
                if key in bad_c_for_pair:
                    if x in bad_c_for_pair[key]:
                        return False
        return True

    def can_add_fast(selected_set, x):
        """Faster check using precomputed data."""
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i + 1, len(sel_list)):
                b = sel_list[j]
                # Check if {a, b, x} is bad
                g_ab = gcd_table[a][b]
                if gcd_table[b][x] == g_ab and gcd_table[x][a] == g_ab:
                    return False
        return True

    def greedy_build(order, use_fast=False):
        """Greedy build following given order."""
        selected = set()
        for x in order:
            if use_fast:
                ok = can_add_fast(selected, x)
            else:
                ok = can_add(selected, x)
            if ok:
                selected.add(x)
        return selected

    def local_search(selected_set, time_limit=3.0):
        """Local search: try to add elements, then try swaps."""
        selected = set(selected_set)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if x not in selected:
                    if can_add(selected, x):
                        selected.add(x)
                        improved = True
                        break
            # If we can't add, try swap: remove one, add another
            if not improved:
                for x in list(selected):
                    remaining = selected - {x}
                    for y in range(1, N + 1):
                        if y not in selected:
                            if can_add(remaining, y):
                                selected = remaining | {y}
                                improved = True
                                break
                    if improved:
                        break
        return selected

    def local_search_v2(selected_set, time_limit=3.0):
        """Enhanced local search with 2-opt style moves."""
        selected = set(selected_set)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Phase 1: Try adding any element
            for x in range(1, N + 1):
                if x not in selected:
                    if can_add(selected, x):
                        selected.add(x)
                        improved = True
                        break
            # Phase 2: Try single swaps
            if not improved:
                for x in list(selected):
                    remaining = selected - {x}
                    for y in range(1, N + 1):
                        if y not in selected:
                            if can_add(remaining, y):
                                selected = remaining | {y}
                                improved = True
                                break
                    if improved:
                        break
            # Phase 3: Try 2-for-1 swaps (remove 2, add 1) - only if size increases
            if not improved and time.time() - start < time_limit:
                for i, x in enumerate(list(selected)):
                    for y in list(selected):
                        if x == y:
                            continue
                        remaining = selected - {x, y}
                        for z in range(1, N + 1):
                            if z not in selected:
                                if can_add(remaining, z):
                                    new_set = remaining | {z}
                                    if len(new_set) > len(selected):
                                        selected = new_set
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
        return selected

    best = set()
    best_size = 0

    # Strategy 1: Natural order greedy
    base_order = list(range(1, N + 1))
    result = greedy_build(base_order)
    result = local_search_v2(result, time_limit=5.0)
    if len(result) > best_size:
        best = set(result)
        best_size = len(result)

    # Strategy 2: Order by number of divisors (fewer divisors = more coprime to others)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order_by_divs = sorted(range(1, N + 1), key=lambda x: num_divisors(x))
    result = greedy_build(order_by_divs)
    result = local_search_v2(result, time_limit=5.0)
    if len(result) > best_size:
        best = set(result)
        best_size = len(result)

    # Strategy 3: Order by value (primes first, then composites)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(1, N + 1) if is_prime(x)]
    composites = [x for x in range(1, N + 1) if not is_prime(x)]
    order_primes_first = primes + composites
    result = greedy_build(order_primes_first)
    result = local_search_v2(result, time_limit=5.0)
    if len(result) > best_size:
        best = set(result)
        best_size = len(result)

    # Strategy 4: Randomized restarts
    import random
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = greedy_build(order)
        result = local_search_v2(result, time_limit=2.0)
        if len(result) > best_size:
            best = set(result)
            best_size = len(result)
        restarts += 1
        if best_size >= 90 and restarts > 50:
            break

    # Final polish: try to add more elements to best
    if time.time() < deadline - 2:
        best = local_search_v2(best, time_limit=3.0)
        best_size = len(best)

    return list(best) if best else [1]
# EVOLVE-BLOCK-END
