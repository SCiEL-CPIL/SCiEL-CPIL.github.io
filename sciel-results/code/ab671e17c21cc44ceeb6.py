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

    # Precompute primes up to 256 (max sum 128+128)
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False

    # We want to find A (3 distinct) and B (3 distinct) such that
    # 1. All a+b are prime (ensures pairwise coprime)
    # 2. All a+b are distinct (maximizes cardinality to 9)
    
    # Strategy:
    # Iterate over candidate A sets (small odds).
    # For each A, find the set of B values such that a+b is prime for all a in A.
    # If there are at least 3 such B values, pick 3 and check if sums are distinct.
    
    odds = list(range(1, 51, 2)) # Search small odds first
    evens = list(range(2, 51, 2)) # Search small evens first
    
    # Also try small evens for A and odds for B, just in case
    # But odd+even=odd is better for primality.
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5 # Fallback score
    
    def score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    best_score = score(best_A, best_B)
    
    found = False
    
    # Optimized Search
    # 1. Pick A from combinations of odds
    # 2. Filter B candidates: b is valid if (a+b) is prime for all a in A
    # 3. If >=3 valid B's, try combinations of 3 B's
    
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > DEADLINE:
            break
        
        A = list(A)
        
        # Find valid B's
        valid_bs = []
        for b in evens:
            valid = True
            for a in A:
                if not is_prime[a + b]:
                    valid = False
                    break
            if valid:
                valid_bs.append(b)
                
        if len(valid_bs) < 3:
            continue
            
        # Try combinations of 3 from valid_bs
        for B in itertools.combinations(valid_bs, 3):
            B = list(B)
            
            # Check if sums are distinct
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
                    
            if len(sums) == 9:
                # Since all sums are prime, they are pairwise coprime
                # (unless one is 2, but odd+even=odd, so no 2s)
                return [A, B]
                
    # If not found in small range, expand search
    odds = list(range(1, 101, 2))
    evens = list(range(2, 101, 2))
    
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > DEADLINE:
            break
            
        A = list(A)
        
        valid_bs = []
        for b in evens:
            valid = True
            for a in A:
                if not is_prime[a + b]:
                    valid = False
                    break
            if valid:
                valid_bs.append(b)
                
        if len(valid_bs) < 3:
            continue
            
        for B in itertools.combinations(valid_bs, 3):
            B = list(B)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
                    
            if len(sums) == 9:
                return [A, B]
                
    # Fallback: Return best found so far
    return [best_A, best_B]
