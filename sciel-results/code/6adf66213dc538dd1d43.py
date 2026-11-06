"""Direct construction: find A, B with 9 distinct pairwise-coprime sums (all primes)."""
import math
import time
import itertools

def _is_prime(n):
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

def _pairwise_coprime(vals):
    """Check all pairs are coprime. Returns True if valid."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    # Strategy: find A (3 odds) and B (3 evens) such that all 9 sums are prime.
    # This guarantees pairwise coprimality. For 9 distinct sums, need disjoint diff sets.
    # 
    # Efficient approach: For each A, find B values where all a+b are prime.
    # Then pick 3 such B values with disjoint difference sets from A.
    
    # Phase 1: Small range, all-prime sums
    for max_val in [40, 60, 80, 128]:
        odds = list(range(1, max_val + 1, 2))
        evens = list(range(2, max_val + 1, 2))
        
        for A in itertools.combinations(odds, 3):
            if time.time() - t_start > DEADLINE:
                break
            A = list(A)
            a_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    a_diffs.add(A[j] - A[i])
            
            # Find B values where all a+b are prime
            valid_b = []
            for b in evens:
                if all(is_prime[a + b] for a in A):
                    valid_b.append(b)
            
            if len(valid_b) < 3:
                continue
            
            # Find 3 from valid_b with disjoint difference sets
            for B in itertools.combinations(valid_b, 3):
                if time.time() - t_start > DEADLINE:
                    break
                B = list(B)
                b_diffs = set()
                for i in range(3):
                    for j in range(i+1, 3):
                        b_diffs.add(B[j] - B[i])
                
                if a_diffs & b_diffs:
                    continue
                
                # All 9 sums are prime and distinct -> valid
                return [A, B]
    
    # Phase 2: Search for pairwise coprime (not necessarily all prime)
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                a_diffs.add(A[j] - A[i])
        
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    b_diffs.add(B[j] - B[i])
            
            if a_diffs & b_diffs:
                continue
            
            distinct_sums = list(set(a + b for a in A for b in B))
            if len(distinct_sums) == 9 and _pairwise_coprime(distinct_sums):
                return [A, B]
    
    # Phase 3: Try all numbers (mixed parity)
    all_nums = list(range(1, 129))
    for A in itertools.combinations(all_nums, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                a_diffs.add(A[j] - A[i])
        
        for B in itertools.combinations(all_nums, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    b_diffs.add(B[j] - B[i])
            
            if a_diffs & b_diffs:
                continue
            
            distinct_sums = list(set(a + b for a in A for b in B))
            if len(distinct_sums) == 9 and _pairwise_coprime(distinct_sums):
                return [A, B]
    
    # Absolute fallback
    return [1, 3, 7], [2, 4, 8]
