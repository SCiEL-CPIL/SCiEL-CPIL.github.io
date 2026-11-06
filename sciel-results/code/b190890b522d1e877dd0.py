import itertools
import math
import time

def construct():
    t_start = time.time()
    TIME_LIMIT = 85
    
    MAX_SUM = 256
    # Sieve for primes
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
                
    # Precompute prime factors
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    prime_factors = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        n = i
        while n > 1:
            p = spf[n]
            prime_factors[i].add(p)
            while n % p == 0:
                n //= p
                
    def check_pairwise_coprime(vals):
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if prime_factors[vals[i]] & prime_factors[vals[j]]:
                    return False
        return True

    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]

    # Search small ranges first
    for max_val in [20, 30, 50, 80, 128]:
        if time.time() - t_start > TIME_LIMIT:
            break
        range_vals = list(range(1, max_val + 1))
        A_combos = list(itertools.combinations(range_vals, 3))
        B_combos = list(itertools.combinations(range_vals, 3))
        
        for A in A_combos:
            if time.time() - t_start > TIME_LIMIT:
                break
            A = list(A)
            # Calculate difference set of A
            diff_A = set()
            for i in range(3):
                for j in range(3):
                    diff_A.add(A[i] - A[j])
            
            for B in B_combos:
                B = list(B)
                # Calculate difference set of B
                diff_B = set()
                for i in range(3):
                    for j in range(3):
                        diff_B.add(B[i] - B[j])
                
                # Check distinctness
                if diff_A & diff_B != {0}:
                    continue
                    
                sums = [a + b for a in A for b in B]
                distinct_sums = list(set(sums))
                
                if len(distinct_sums) <= best_score:
                    continue
                    
                if check_pairwise_coprime(distinct_sums):
                    if len(distinct_sums) > best_score:
                        best_score = len(distinct_sums)
                        best_A = A
                        best_B = B
                        if best_score == 9:
                            return [best_A, best_B]
                            
    return [best_A, best_B]
