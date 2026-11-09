"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin (90s limit)
    
    N = 128
    # Precompute GCD table for fast access
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad[a][b] = set of x such that (a,b,x) is a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for x in range(1, N + 1):
                if x == a or x == b:
                    continue
                if gcd_table[a][x] == gab and gcd_table[b][x] == gab:
                    bad[a][b].add(x)
                    bad[b][a].add(x)

    # Precompute conflict_count[x] = number of pairs (a,b) in [1..N] where x is bad
    # This helps prioritize which elements to add first
    conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for x in bad[a][b]:
                conflict_count[x] += 1

    def can_add(selected, x):
        """
        Check if x can be added to selected without creating a bad triple.
        A bad triple (a, b, x) exists if gcd(a,b) == gcd(a,x) == gcd(b,x).
        """
        n = len(selected)
        # Iterate over pairs in selected
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                if x in bad[a][b]:
                    return False
        return True

    def greedy_build(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected

    def local_search_fill(selected):
        """
        Try to add any remaining element that fits.
        This is a local search with 'add' moves.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        changed = True
        while changed:
            changed = False
            for x in all_nums - selected_set:
                if can_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    changed = True
        return selected

    def local_search_swap(selected):
        """
        Try swap moves to escape local optima.
        Supports: 1-for-1, 2-for-1, 1-for-2 swaps.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        candidates = list(all_nums - selected_set)
        best_size = len(selected)
        
        max_iterations = 500
        it = 0
        
        while it < max_iterations and time.time() < deadline - 3:
            it += 1
            improved = False
            
            # Try 1-for-1 swaps
            for a in selected:
                if time.time() > deadline - 3:
                    break
                for b in candidates:
                    if b in selected_set:
                        continue
                    temp_selected = [x for x in selected if x != a]
                    if can_add(temp_selected, b):
                        new_selected = temp_selected + [b]
                        new_selected = local_search_fill(new_selected)
                        if len(new_selected) > best_size:
                            selected = new_selected
                            selected_set = set(selected)
                            candidates = list(all_nums - selected_set)
                            best_size = len(selected)
                            improved = True
                            break
                if improved or time.time() > deadline - 3:
                    break
            
            if improved or time.time() > deadline - 3:
                continue
            
            # Try 2-for-1 swaps (remove 2, add 1)
            n_sel = len(selected)
            for i in range(n_sel):
                if time.time() > deadline - 3:
                    break
                a = selected[i]
                for j in range(i + 1, n_sel):
                    b = selected[j]
                    for c in candidates:
                        if c in selected_set:
                            continue
                        temp_selected = [x for k, x in enumerate(selected) if k != i and k != j]
                        if can_add(temp_selected, c):
                            new_selected = temp_selected + [c]
                            new_selected = local_search_fill(new_selected)
                            if len(new_selected) > best_size:
                                selected = new_selected
                                selected_set = set(selected)
                                candidates = list(all_nums - selected_set)
                                best_size = len(selected)
                                improved = True
                                break
                    if improved or time.time() > deadline - 3:
                        break
                if improved or time.time() > deadline - 3:
                    break
            
            if improved or time.time() > deadline - 3:
                continue
            
            # Try 1-for-2 swaps (remove 1, add 2)
            for a in selected:
                if time.time() > deadline - 3:
                    break
                for b in candidates:
                    if b in selected_set:
                        continue
                    temp_selected = [x for x in selected if x != a]
                    if can_add(temp_selected, b):
                        # Now try to add one more
                        for c in candidates:
                            if c in selected_set or c == b:
                                continue
                            if can_add(temp_selected + [b], c):
                                new_selected = temp_selected + [b, c]
                                new_selected = local_search_fill(new_selected)
                                if len(new_selected) > best_size:
                                    selected = new_selected
                                    selected_set = set(selected)
                                    candidates = list(all_nums - selected_set)
                                    best_size = len(selected)
                                    improved = True
                                    break
                    if improved or time.time() > deadline - 3:
                        break
                if improved or time.time() > deadline - 3:
                    break
            
            if not improved:
                break
            
        return selected

    nums = list(range(1, N + 1))
    best = []

    # Heuristic orderings
    orderings = []
    
    # 1. Ascending
    orderings.append(nums[:])
    # 2. Descending
    orderings.append(nums[::-1])
    
    # 3. By number of divisors (descending)
    def div_count(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    orderings.append(sorted(nums, key=lambda x: -div_count(x)))
    orderings.append(sorted(nums, key=lambda x: div_count(x)))
    
    # 4. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)
    
    # 5. By conflict count (ascending - less conflicting first)
    orderings.append(sorted(nums, key=lambda x: conflict_count[x]))
    orderings.append(sorted(nums, key=lambda x: -conflict_count[x]))

    # 6. Random restarts with varied strategies
    rng = random.Random(42)
    
    # More random restarts
    for _ in range(60):
        o = nums[:]
        rng.shuffle(o)
        orderings.append(o)
    
    # Some with bias toward low-conflict elements
    for _ in range(20):
        o = sorted(nums, key=lambda x: conflict_count[x] * 0.5 + rng.random() * 50)
        orderings.append(o)

    for order in orderings:
        if time.time() > deadline - 15:
            break
        
        # Greedy build
        cand = greedy_build(order)
        
        # Local search fill (add any possible)
        cand = local_search_fill(cand)
        
        # Local search swap (try to improve if stuck)
        if time.time() < deadline - 5:
            cand = local_search_swap(cand)
        
        if len(cand) > len(best):
            best = cand
            
    # Final polish: try to add any remaining elements
    if best and time.time() < deadline - 2:
        best = local_search_fill(best)
    
    # Ensure non-empty
    if not best:
        best = [1]
        
    return best
# EVOLVE-BLOCK-END
