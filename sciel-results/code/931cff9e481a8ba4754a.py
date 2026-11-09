import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def is_valid_addition(S, x, S_set=None):
        """
        Check if x can be added to S.
        S is a list of ints.
        S_set is optional precomputed set for speed.
        """
        if S_set is None:
            S_set = set(S)
        
        if x in S_set:
            return False
        
        # Check 1 & 2: a | x*c and a | b*x
        for a in S:
            g = gcd(a, x)
            k = a // g
            if k == 1:
                # a divides x.
                # Then a divides x*c for ALL c.
                # So if there is any c in S, c != a, it's a violation.
                if len(S) > 1:
                    return False
                continue
            
            # Check if any c in S (c != a) is a multiple of k
            for c in S:
                if c == a:
                    continue
                if c % k == 0:
                    return False
        
        # Check 3: x | b*c
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(i + 1, n):
                c = S[j]
                if (b * c) % x == 0:
                    return False
        
        return True

    def local_search(S, max_iters=200):
        """Try to improve S by swapping or adding elements."""
        S_set = set(S)
        candidates = [x for x in range(2, 257) if x not in S_set]
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding
            for x in candidates:
                if is_valid_addition(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one, add one
            for i, elem in enumerate(S):
                if time.time() > deadline:
                    return S
                S.remove(elem)
                S_set.discard(elem)
                S_set_copy = set(S_set)
                for x in candidates:
                    if is_valid_addition(S, x, S_set_copy):
                        S.append(x)
                        S_set.add(x)
                        candidates.remove(x)
                        candidates.append(elem)
                        improved = True
                        break
                if not improved:
                    S.append(elem)
                    S_set.add(elem)
        return S

    def greedy(order, use_local_search=True):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S_set:
                if is_valid_addition(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
        if use_local_search and time.time() < deadline:
            S = local_search(S)
        return S

    best_set = []
    
    # Precompute primes
    primes = [a for a in range(2, 257) if all(a % d for d in range(2, int(a**0.5) + 1))]
    composites = [a for a in range(2, 257) if a not in primes]
    all_nums = list(range(2, 257))
    
    # Strategy 1: Descending order
    order1 = list(range(256, 1, -1))
    S1 = greedy(order1)
    if len(S1) > len(best_set):
        best_set = S1
        
    # Strategy 2: Primes first, then composites descending
    order2 = primes + sorted(composites, reverse=True)
    S2 = greedy(order2)
    if len(S2) > len(best_set):
        best_set = S2
        
    # Strategy 3: Random shuffles with local search
    random.seed(42)
    for i in range(30): # Try 30 random orders
        if time.time() > deadline:
            break
        order = all_nums[:]
        random.shuffle(order)
        Sr = greedy(order)
        if len(Sr) > len(best_set):
            best_set = Sr
            
    # Strategy 4: Sort by number of divisors (ascending)
    order4 = sorted(all_nums, key=lambda x: len(get_divisors(x)))
    S4 = greedy(order4)
    if len(S4) > len(best_set):
        best_set = S4
        
    # Strategy 5: Primes + select composites with few small prime factors
    # Composites that are products of large primes are easier to include
    def num_small_factors(n):
        count = 0
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                count += 1
        return count
    
    order5 = primes + sorted(composites, key=lambda x: (num_small_factors(x), -x))
    S5 = greedy(order5)
    if len(S5) > len(best_set):
        best_set = S5
        
    # Final local search on best set
    if time.time() < deadline:
        best_set = local_search(best_set, max_iters=500)
        
    # Convert to list of ints
    return [int(x) for x in best_set]
