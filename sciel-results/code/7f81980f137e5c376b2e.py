"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
    
    # Precompute all bad triples as a set for O(1) lookup
    # bad_triples[a][b] contains set of c such that (a,b,c) is a bad triple
    bad_triples = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples[a][b].add(c)
                    bad_triples[b][a].add(c)
                    bad_triples[a][c].add(b)
                    bad_triples[c][a].add(b)
                    bad_triples[b][c].add(a)
                    bad_triples[c][b].add(a)
    
    def is_compatible_fast(x, selected):
        """Check if x can be added to selected without creating a bad triple (fast)."""
        for a in selected:
            if x in bad_triples[a][x] or (a in bad_triples[x][a]):
                # Check if there's a c in selected such that (a, x, c) is bad
                # i.e., c is in bad_triples[a][x]
                for c in selected:
                    if c != a and c != x and c in bad_triples[a][x]:
                        return False
        return True
    
    def is_compatible(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for a, b in combinations(selected, 2):
            if a in bad_triples[b][x] or b in bad_triples[a][x]:
                # More directly: check if (a,b,x) forms a bad triple
                # This means lcm(a,b) == lcm(a,x) == lcm(b,x)
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if compatible."""
        selected = []
        selected_set = set()
        for x in order:
            compatible = True
            for a in selected:
                for b in selected:
                    if a >= b:
                        continue
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                selected.append(x)
                selected_set.add(x)
        return selected
    
    def local_improve(selected):
        """Try to improve by adding elements or swapping."""
        selected = list(selected)
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            # Try adding any missing element
            in_set = set(selected)
            for x in range(1, n + 1):
                if x not in in_set:
                    compatible = True
                    for a in selected:
                        for b in selected:
                            if a >= b:
                                continue
                            if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        selected.append(x)
                        in_set.add(x)
                        improved = True
                        if time.time() > deadline - 3:
                            return selected
            # Try 1-for-2 swaps: remove one element, try to add two
            if time.time() > deadline - 3:
                break
            for i in range(len(selected)):
                if time.time() > deadline - 3:
                    break
                a = selected[i]
                temp = selected[:i] + selected[i+1:]
                temp_set = set(temp)
                added = 0
                for x in range(1, n + 1):
                    if x not in temp_set:
                        compatible = True
                        for p in temp:
                            for q in temp:
                                if p >= q:
                                    continue
                                if lcm_table[p][q] == lcm_table[p][x] == lcm_table[q][x]:
                                    compatible = False
                                    break
                            if not compatible:
                                break
                        if compatible:
                            temp.append(x)
                            temp_set.add(x)
                            added += 1
                            if added >= 2:
                                break
                if len(temp) > len(selected):
                    selected = temp
                    improved = True
                    break
        return selected
    
    # Try multiple orderings
    best = []
    
    # Order 1: natural order
    if time.time() < deadline - 5:
        result = greedy_solve(numbers)
        if len(result) > len(best):
            best = result
    
    # Order 2: reverse
    if time.time() < deadline - 5:
        result = greedy_solve(list(reversed(numbers)))
        if len(result) > len(best):
            best = result
    
    # Order 3: by number of divisors (fewer first)
    if time.time() < deadline - 5:
        def num_divisors(x):
            count = 0
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    count += 1
                    if d != x // d:
                        count += 1
            return count
        order = sorted(numbers, key=num_divisors)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 4: by number of distinct prime factors
    if time.time() < deadline - 5:
        def num_distinct_primes(x):
            count = 0
            d = 2
            while d * d <= x:
                if x % d == 0:
                    count += 1
                    while x % d == 0:
                        x //= d
                d += 1
            if x > 1:
                count += 1
            return count
        order = sorted(numbers, key=num_distinct_primes)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Order 5: by sum of LCMs with all others (more isolated first)
    if time.time() < deadline - 5:
        def lcm_complexity(x):
            total = 0
            for y in range(1, n + 1):
                if y != x:
                    total += lcm_table[x][y]
            return total
        order = sorted(numbers, key=lcm_complexity)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
    
    # Local improvement on best
    if time.time() < deadline - 5:
        best = local_improve(best)
    
    # Try random orderings with local improvement
    import random
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        order = list(numbers)
        random.shuffle(order)
        result = greedy_solve(order)
        if len(result) > len(best):
            best = result
        elif len(result) >= len(best) - 1:
            # Also improve near-best solutions
            improved = local_improve(result)
            if len(improved) > len(best):
                best = improved
    
    # Final local improvement
    if time.time() < deadline - 3:
        best = local_improve(best)
    
    return best
# EVOLVE-BLOCK-END
