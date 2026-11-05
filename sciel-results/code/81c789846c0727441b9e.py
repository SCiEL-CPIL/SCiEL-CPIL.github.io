"""Rewritten program: targeted search for 9-element pairwise-coprime sumset."""
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute prime factor sets for odd numbers up to 254
    def prime_factors(n):
        factors = set()
        d = 3
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 2
        if n > 1:
            factors.add(n)
        return factors
    
    pf_cache = {}
    for n in range(3, 255, 2):
        pf_cache[n] = prime_factors(n)
    
    def coprime(a, b):
        return len(pf_cache[a] & pf_cache[b]) == 0
    
    # Odd numbers in [1, 127]
    odds = list(range(1, 128, 2))
    # Even numbers in [2, 128]
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = None
    best_B = None
    
    def try_search(A_candidates, B_candidates):
        nonlocal best_count, best_A, best_B
        for A in itertools.combinations(A_candidates, 3):
            if time.time() > deadline:
                return
            a1, a2, a3 = A
            
            # Find all valid b values
            valid_bs = []
            for b in B_candidates:
                s1 = a1 + b
                s2 = a2 + b
                s3 = a3 + b
                if coprime(s1, s2) and coprime(s1, s3) and coprime(s2, s3):
                    valid_bs.append(b)
            
            if len(valid_bs) < 3:
                continue
            
            for B in itertools.combinations(valid_bs, 3):
                if time.time() > deadline:
                    return
                b1, b2, b3 = B
                
                sums = [
                    a1 + b1, a1 + b2, a1 + b3,
                    a2 + b1, a2 + b2, a2 + b3,
                    a3 + b1, a3 + b2, a3 + b3
                ]
                
                distinct_sums = list(set(sums))
                count = len(distinct_sums)
                
                if count <= best_count:
                    continue
                
                # Check all pairwise coprime
                ok = True
                for i in range(len(distinct_sums)):
                    for j in range(i + 1, len(distinct_sums)):
                        if not coprime(distinct_sums[i], distinct_sums[j]):
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    best_count = count
                    best_A = list(A)
                    best_B = list(B)
                    
                    if best_count == 9:
                        return
    
    # Case 1: A all odd, B all even
    try_search(odds, evens)
    
    # Case 2: A all even, B all odd
    if best_count < 9 and time.time() < deadline:
        try_search(evens, odds)
    
    if best_A is not None:
        return [best_A, best_B]
    else:
        return [[1, 7, 13], [4, 10, 16]]
