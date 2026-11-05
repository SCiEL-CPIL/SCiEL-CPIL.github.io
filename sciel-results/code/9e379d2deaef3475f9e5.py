import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 88  # 88s to leave margin for safety
    
    # Precompute gcd table for all pairs in [1, 128]
    n_max = 128
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i + 1, n_max + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all three pairwise gcds equal."""
        g1 = gcd_table[a][b]
        g2 = gcd_table[a][c]
        g3 = gcd_table[b][c]
        return g1 == g2 == g3
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                g_ab = gcd_table[a][b]
                g_ax = gcd_table[a][x]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    return False
        return True
    
    def greedy_build(order):
        """Greedy: add elements in given order if they don't create a bad triple."""
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def local_search(selected, time_limit):
        """Try to improve the set by adds and swaps."""
        selected = list(selected)
        remaining = [x for x in range(1, 129) if x not in selected]
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try adding a single element
            for idx, x in enumerate(remaining):
                if can_add(x, selected):
                    selected.append(x)
                    remaining.pop(idx)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add another
                found_swap = False
                n_sel = len(selected)
                for i in range(n_sel):
                    y = selected[i]
                    candidate = selected[:i] + selected[i+1:]
                    for j, x in enumerate(remaining):
                        if can_add(x, candidate):
                            selected = candidate + [x]
                            remaining.pop(j)
                            found_swap = True
                            break
                    if found_swap:
                        break
                if not found_swap:
                    break
            
            # Try 2-swap: remove two, add two
            if not improved and time.time() - start < time_limit - 2:
                found_2swap = False
                n_sel = len(selected)
                for i in range(n_sel):
                    for j in range(i + 1, n_sel):
                        candidate = selected[:i] + selected[i+1:j] + selected[j+1:]
                        for x1 in remaining:
                            if can_add(x1, candidate):
                                temp_sel = candidate + [x1]
                                for x2 in remaining:
                                    if x2 != x1 and can_add(x2, temp_sel):
                                        selected = temp_sel + [x2]
                                        # Remove x1 and x2 from remaining
                                        rem_idx1 = remaining.index(x1)
                                        rem_idx2 = remaining.index(x2)
                                        if rem_idx1 < rem_idx2:
                                            remaining.pop(rem_idx2)
                                            remaining.pop(rem_idx1)
                                        else:
                                            remaining.pop(rem_idx1)
                                            remaining.pop(rem_idx2)
                                        found_2swap = True
                                        break
                                if found_2swap:
                                    break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if not found_2swap:
                    break
        
        return selected
    
    # Helper: count distinct prime factors
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
    
    # Helper: largest prime factor
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
    
    # Helper: total number of prime factors (with multiplicity)
    def total_prime_factors(n):
        count = 0
        d = 2
        m = n
        while d * d <= m:
            while m % d == 0:
                count += 1
                m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    # Multiple greedy strategies
    strategies = []
    
    # Strategy 1: Natural order
    strategies.append(list(range(1, 129)))
    
    # Strategy 2: Reverse order
    strategies.append(list(range(128, 0, -1)))
    
    # Strategy 3: Powers of 2 first, then rest
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
    rest = [x for x in range(1, 129) if x not in powers_of_2]
    strategies.append(powers_of_2 + rest)
    
    # Strategy 4: Order by number of distinct prime factors (ascending)
    strategies.append(sorted(range(1, 129), key=num_distinct_primes))
    
    # Strategy 5: Order by largest prime factor (descending)
    strategies.append(sorted(range(1, 129), key=largest_prime_factor, reverse=True))
    
    # Strategy 6: Order by total prime factors (ascending)
    strategies.append(sorted(range(1, 129), key=total_prime_factors))
    
    # Strategy 7: Primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    primes = [x for x in range(1, 129) if is_prime(x)]
    composites = [x for x in range(1, 129) if not is_prime(x)]
    strategies.append(primes + composites)
    
    # Strategy 8: Order by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    strategies.append(sorted(range(1, 129), key=num_divisors))
    
    # Run all greedy strategies and keep the best
    best = []
    for order in strategies:
        result = greedy_build(order)
        if len(result) > len(best):
            best = result
    
    # Local search to improve
    if time.time() < deadline - 15:
        best = local_search(best, deadline - time.time() - 5)
    
    # Try local search from different starting points if time permits
    if time.time() < deadline - 30:
        for order in strategies[:4]:  # Try first 4 strategies
            if time.time() < deadline - 15:
                result = greedy_build(order)
                result = local_search(result, min(10, deadline - time.time() - 5))
                if len(result) > len(best):
                    best = result
    
    # Final local search if time permits
    if time.time() < deadline - 5:
        best = local_search(best, deadline - time.time() - 2)
    
    return best
