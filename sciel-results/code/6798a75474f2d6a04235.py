import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    N = 128
    all_elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[(a, b)] = a * b // gcd(a, b)
    
    def lcm(a, b):
        if a > b:
            a, b = b, a
        return lcm_table[(a, b)]
    
    # Precompute bad triples: for each x, list of pairs (a,b) such that {x,a,b} is bad
    # conflict_pairs[x] = list of (a,b) with a < b, a != x, b != x
    # such that lcm(x,a) == lcm(x,b) == lcm(a,b)
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    # {a, b, c} is a bad triple
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (a, b) in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True
    
    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    # Local search
    def local_search(S, deadline):
        S = set(S)
        full_set = all_elements
        no_improve_count = 0
        
        while time.time() < deadline - 1.0:
            improved = False
            
            # Phase 1: Try to add any element
            for x in full_set:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # Phase 2: Try 1-swaps
            found_swap = False
            S_list = list(S)
            for a in S_list:
                S.remove(a)
                for x in full_set:
                    if x not in S and can_add(x, S):
                        S.add(x)
                        found_swap = True
                        break
                if not found_swap:
                    S.add(a)
                if found_swap:
                    break
            
            if found_swap:
                no_improve_count = 0
                continue
            
            # Phase 3: Try 2-swaps
            if time.time() < deadline - 3.0:
                found_2swap = False
                S_list = list(S)
                n = len(S_list)
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = S_list[i], S_list[j]
                        S.remove(a)
                        S.remove(b)
                        # Try to add two elements
                        added_first = False
                        for x in full_set:
                            if x in S or not can_add(x, S):
                                continue
                            S.add(x)
                            added_first = True
                            for y in full_set:
                                if y in S or y == x or not can_add(y, S):
                                    continue
                                S.add(y)
                                found_2swap = True
                                break
                            if found_2swap:
                                break
                            else:
                                S.remove(x)
                        if found_2swap:
                            break
                        if added_first:
                            S.remove(x)
                        if not found_2swap:
                            S.add(a)
                            S.add(b)
                    if found_2swap:
                        break
                
                if found_2swap:
                    no_improve_count = 0
                    continue
            
            # No improvement in this round
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        
        return S
    
    # Helper functions for ordering
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    def max_prime_factor(n):
        max_pf = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                max_pf = d
                n //= d
            d += 1
        if n > 1:
            max_pf = n
        return max_pf
    
    def radical(n):
        """Product of distinct prime factors."""
        rad = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                rad *= d
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            rad *= n
        return rad
    
    # Build orderings
    primes = [x for x in all_elements if is_prime(x)]
    non_primes = [x for x in all_elements if not is_prime(x)]
    
    orderings = []
    # 1. Primes first
    orderings.append(primes + non_primes)
    # 2. By number of divisors (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: num_divisors(x)))
    # 3. Reverse
    orderings.append(list(reversed(all_elements)))
    # 4. Forward
    orderings.append(all_elements)
    # 5. By max prime factor (larger first)
    orderings.append(sorted(all_elements, key=lambda x: -max_prime_factor(x)))
    # 6. By max prime factor (smaller first)
    orderings.append(sorted(all_elements, key=lambda x: max_prime_factor(x)))
    # 7. By radical (larger first)
    orderings.append(sorted(all_elements, key=lambda x: -radical(x)))
    # 8. By radical (smaller first)
    orderings.append(sorted(all_elements, key=lambda x: radical(x)))
    # 9. By number of distinct prime factors (more first)
    def num_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    orderings.append(sorted(all_elements, key=lambda x: -num_distinct_pf(x)))
    # 10. By num distinct pf (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: num_distinct_pf(x)))
    
    # Random orderings
    random.seed(42)
    for i in range(30):
        orderings.append(random.sample(all_elements, N))
    
    best_set = set()
    best_size = 0
    
    # Run greedy with all orderings
    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # More random restarts
    while time.time() < deadline - 15.0:
        candidates = random.sample(all_elements, N)
        S = greedy(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on best
    if time.time() < deadline - 5.0:
        improved = local_search(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Try local search on a few other good sets too
    if time.time() < deadline - 5.0:
        # Re-run a few top greedy results and local search them
        for order in orderings[:5]:
            if time.time() > deadline - 3.0:
                break
            S = greedy(order)
            if len(S) >= best_size - 2:
                improved = local_search(S, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved
    
    # Final validation
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            valid = False
            break
    
    if not valid:
        best_set = {1}
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
