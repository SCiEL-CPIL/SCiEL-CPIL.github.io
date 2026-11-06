"""Constructs an integer n <= 10^18 maximizing tau_perp(n)."""

import time
from math import gcd
from itertools import combinations

def _tau_perp_squarefree(primes):
    """Compute tau_perp for a square-free number given its prime list."""
    k = len(primes)
    if k == 0:
        return 0
    # Generate all divisors as (value, bitmask)
    # Divisor d has bitmask m where bit j is set if primes[j] divides d
    # Two divisors are coprime iff their bitmasks are disjoint
    divs = [(1, 0)]
    for i, p in enumerate(primes):
        new_divs = []
        bit = 1 << i
        for val, mask in divs:
            new_divs.append((val, mask))
            new_divs.append((val * p, mask | bit))
        divs = new_divs
    
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divs) - 1):
        if divs[i][1] & divs[i+1][1] == 0:
            count += 1
    return count

def _tau_perp_general(n, primes, exps):
    """Compute tau_perp for n given its prime factorization."""
    k = len(primes)
    # Generate all divisors
    divs = [1]
    for i, (p, e) in enumerate(zip(primes, exps)):
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    
    if all(e == 1 for e in exps):
        # Square-free: use bitmask
        divisor_mask = {}
        for d in divs:
            mask = 0
            for j, p in enumerate(primes):
                if d % p == 0:
                    mask |= (1 << j)
            divisor_mask[d] = mask
        count = 0
        for i in range(len(divs) - 1):
            if divisor_mask[divs[i]] & divisor_mask[divs[i+1]] == 0:
                count += 1
        return count
    else:
        count = 0
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i + 1]) == 1:
                count += 1
        return count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 2
    best_score = 0
    
    # First 15 primes
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def make_n(primes, exps):
        n = 1
        for p, e in zip(primes, exps):
            n *= (p ** e)
        return n
    
    def try_candidate(n, primes, exps):
        nonlocal best_n, best_score
        if n < 2 or n > 10**18:
            return
        if time.time() > deadline - 3:
            return
        if all(e == 1 for e in exps):
            s = _tau_perp_squarefree(primes)
        else:
            s = _tau_perp_general(n, primes, exps)
        if s > best_score:
            best_score = s
            best_n = n
    
    # === Phase 1: Try all subsets of primes (square-free) ===
    # For k primes, n = product of those primes
    # Check which subsets fit in 10^18
    
    # Precompute products of first m primes
    prefix_prod = [1] * 16
    for i in range(15):
        prefix_prod[i+1] = prefix_prod[i] * all_primes[i]
    
    # Maximum number of primes we can use
    max_k = 0
    for k in range(1, 16):
        if prefix_prod[k] <= 10**18:
            max_k = k
        else:
            break
    
    # Try all subsets of size max_k, max_k-1, etc.
    # Start with the largest subsets
    
    for k in range(max_k, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # For k = max_k, only the first k primes might work
        # For smaller k, try various combinations
        if k == max_k:
            # Only the first k primes
            subset = all_primes[:k]
            n = make_n(subset, [1]*k)
            if n <= 10**18:
                try_candidate(n, subset, [1]*k)
        else:
            # Try subsets of size k from the first (max_k) primes
            # But this could be C(15, k) which is large for k=10
            # Focus on subsets that include small primes
            if k >= 12:
                # Only try subsets from the first 13 primes
                pool = all_primes[:13]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 5:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        try_candidate(n, list(subset), [1]*k)
            elif k >= 10:
                # Try subsets from the first 12 primes
                pool = all_primes[:12]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 5:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        try_candidate(n, list(subset), [1]*k)
            else:
                # For smaller k, try all subsets from first 11 primes
                pool = all_primes[:11]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 5:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        try_candidate(n, list(subset), [1]*k)
    
    # === Phase 2: Add exponents to the best square-free candidates ===
    # Take the top candidates and try adding exponents to small primes
    
    # Re-evaluate: find the best square-free n, then try exponent variations
    # Focus on the first few primes for exponent increases
    
    # Get the best square-free primes set
    # Re-run to find best
    best_sqfree_n = 2
    best_sqfree_score = 0
    best_sqfree_primes = []
    
    for k in range(max_k, 0, -1):
        if time.time() > deadline - 10:
            break
        if k == max_k:
            subset = all_primes[:k]
            n = make_n(subset, [1]*k)
            if n <= 10**18:
                s = _tau_perp_squarefree(subset)
                if s > best_sqfree_score:
                    best_sqfree_score = s
                    best_sqfree_n = n
                    best_sqfree_primes = subset
        else:
            if k >= 12:
                pool = all_primes[:13]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 10:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        s = _tau_perp_squarefree(list(subset))
                        if s > best_sqfree_score:
                            best_sqfree_score = s
                            best_sqfree_n = n
                            best_sqfree_primes = list(subset)
            elif k >= 10:
                pool = all_primes[:12]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 10:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        s = _tau_perp_squarefree(list(subset))
                        if s > best_sqfree_score:
                            best_sqfree_score = s
                            best_sqfree_n = n
                            best_sqfree_primes = list(subset)
            else:
                pool = all_primes[:11]
                for subset in combinations(pool, k):
                    if time.time() > deadline - 10:
                        break
                    n = 1
                    for p in subset:
                        n *= p
                    if n <= 10**18:
                        s = _tau_perp_squarefree(list(subset))
                        if s > best_sqfree_score:
                            best_sqfree_score = s
                            best_sqfree_n = n
                            best_sqfree_primes = list(subset)
    
    # Now try adding exponents to the best square-free candidate
    if best_sqfree_primes and time.time() < deadline - 15:
        base_primes = best_sqfree_primes
        base_n = best_sqfree_n
        base_score = best_sqfree_score
        
        # Try increasing exponents of the smallest primes
        # n = base_n * 2^a * 3^b * 5^c * ...
        # Budget: 10^18 / base_n
        budget = 10**18 // base_n
        
        # Try small exponent combinations
        # Focus on 2, 3, 5 (the smallest primes in the set)
        # Find indices of 2, 3, 5 in base_primes
        idx_map = {}
        for i, p in enumerate(base_primes):
            idx_map[p] = i
        
        # Try exponent variations for the first 3-4 primes
        small_primes_in_set = []
        for p in [2, 3, 5, 7, 11]:
            if p in idx_map:
                small_primes_in_set.append(p)
        
        if small_primes_in_set:
            # Try all combinations of exponents for these primes
            # Each exponent from 0 to max_e where product fits budget
            # Use DFS
            def dfs_exponents(idx, current_mult, current_exps):
                nonlocal best_n, best_score
                if time.time() > deadline - 5:
                    return
                if idx == len(small_primes_in_set):
                    # Check if total fits
                    if current_mult <= budget:
                        exps = [1] * len(base_primes)
                        for i, p in enumerate(small_primes_in_set):
                            exps[idx_map[p]] = 1 + current_exps[i]
                        n_try = make_n(base_primes, exps)
                        if n_try <= 10**18:
                            try_candidate(n_try, base_primes, exps)
                    return
                
                p = small_primes_in_set[idx]
                # Max exponent for this prime
                max_e = 0
                temp = current_mult
                while temp * p <= budget:
                    temp *= p
                    max_e += 1
                
                for e in range(max_e + 1):
                    dfs_exponents(idx + 1, current_mult * (p ** e), current_exps + [e])
            
            dfs_exponents(0, 1, [])
    
    # === Phase 3: Local hill-climbing on the best candidate ===
    if time.time() < deadline - 10:
        # Try to improve best_n by local changes
        # Factorize best_n
        temp = best_n
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        primes_list = sorted(factors.keys())
        exps_list = [factors[p] for p in primes_list]
        
        # Try incrementing/decrementing exponents
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, p in enumerate(primes_list):
                # Try increasing exponent
                if exps_list[i] < 20:
                    new_exps = exps_list.copy()
                    new_exps[i] += 1
                    n_try = make_n(primes_list, new_exps)
                    if n_try <= 10**18:
                        s = _tau_perp_general(n_try, primes_list, new_exps)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
                            exps_list = new_exps
                            improved = True
                            break
                # Try decreasing exponent (if > 1)
                if exps_list[i] > 1:
                    new_exps = exps_list.copy()
                    new_exps[i] -= 1
                    n_try = make_n(primes_list, new_exps)
                    if n_try <= 10**18:
                        s = _tau_perp_general(n_try, primes_list, new_exps)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
                            exps_list = new_exps
                            improved = True
                            break
        
        # Try adding a new prime
        for p in all_primes:
            if p not in primes_list and time.time() < deadline - 5:
                new_primes = primes_list + [p]
                new_primes.sort()
                new_exps = exps_list + [1]
                # Reorder exps to match new_primes
                new_exps = [exps_list[primes_list.index(p2)] if p2 in primes_list else 1 for p2 in new_primes]
                n_try = make_n(new_primes, new_exps)
                if n_try <= 10**18:
                    s = _tau_perp_general(n_try, new_primes, new_exps)
                    if s > best_score:
                        best_score = s
                        best_n = n_try
                        break
    
    return best_n
