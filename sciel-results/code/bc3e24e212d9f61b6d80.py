import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 85  # 85s to leave margin
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all three pairwise gcds equal."""
        g1 = gcd(a, b)
        g2 = gcd(a, c)
        g3 = gcd(b, c)
        return g1 == g2 == g3
    
    def count_conflicts(selected):
        """Count number of bad triples in selected set."""
        s = list(selected)
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_bad_triple(s[i], s[j], s[k]):
                        count += 1
        return count
    
    def greedy_build(order):
        """Greedy: add elements in given order if they don't create a bad triple."""
        selected = []
        for x in order:
            if all(not is_bad_triple(a, b, x) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected
    
    def local_search(selected, time_limit):
        """Try to improve the set by swaps."""
        selected = list(selected)
        remaining = [x for x in range(1, 129) if x not in selected]
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            # Try adding a single element
            for x in remaining:
                if all(not is_bad_triple(a, b, x) for a, b in combinations(selected, 2)):
                    selected.append(x)
                    remaining.remove(x)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add another
                found_swap = False
                for i, y in enumerate(selected):
                    candidate = selected[:i] + selected[i+1:]
                    for x in remaining:
                        if all(not is_bad_triple(a, b, x) for a, b in combinations(candidate, 2)):
                            selected = candidate + [x]
                            remaining.remove(x)
                            found_swap = True
                            break
                    if found_swap:
                        break
                if not found_swap:
                    break
        
        return selected
    
    # Strategy 1: Greedy with natural order
    best = greedy_build(range(1, 129))
    
    # Strategy 2: Greedy with reverse order
    best2 = greedy_build(range(128, 0, -1))
    if len(best2) > len(best):
        best = best2
    
    # Strategy 3: Greedy starting with powers of 2, then add others
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
    selected = list(powers_of_2)
    for x in range(1, 129):
        if x not in selected:
            if all(not is_bad_triple(a, b, x) for a, b in combinations(selected, 2)):
                selected.append(x)
    if len(selected) > len(best):
        best = selected
    
    # Local search to improve
    if time.time() < deadline - 10:
        best = local_search(best, deadline - time.time() - 5)
    
    # Try a few more greedy strategies with different orderings
    if time.time() < deadline - 20:
        # Order by number of distinct prime factors (ascending)
        def num_distinct_primes(n):
            count = 0
            d = 2
            m = n
            while d * d <= m:
                if m % d == 0:
                    count += 1
                    while m % d == 0:
                        m //= d
                d += 1
            if m > 1:
                count += 1
            return count
        
        order = sorted(range(1, 129), key=num_distinct_primes)
        best3 = greedy_build(order)
        if len(best3) > len(best):
            best = best3
        
        # Order by value of largest prime factor (descending)
        def largest_prime_factor(n):
            lp = 1
            d = 2
            m = n
            while d * d <= m:
                if m % d == 0:
                    lp = d
                    while m % d == 0:
                        m //= d
                d += 1
            if m > 1:
                lp = m
            return lp
        
        order2 = sorted(range(1, 129), key=largest_prime_factor, reverse=True)
        best4 = greedy_build(order2)
        if len(best4) > len(best):
            best = best4
    
    # Final local search if time permits
    if time.time() < deadline - 5:
        best = local_search(best, deadline - time.time() - 2)
    
    return best
