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
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0
    
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
    
    # Strategy: Search for A (odd) and B (even) such that all 9 sums are prime.
    # Use difference-set disjointness to ensure 9 distinct sums.
    # Then check if all 9 are prime (stronger than pairwise coprime).
    
    # Search A from small odds, B from small evens.
    # For efficiency, iterate over difference patterns.
    
    # A = {a, a+d1, a+d1+d2} where d1, d2 > 0
    # B = {b, b+e1, b+e1+e2} where e1, e2 > 0
    # Difference sets: A has {d1, d2, d1+d2}, B has {e1, e2, e1+e2}
    # Need these to be disjoint.
    
    # Try small difference values first.
    # A differences: pick from small values
    # B differences: pick from small values, disjoint from A's
    
    found = False
    
    # Generate candidate difference pairs for A and B
    # A: (d1, d2) with d1 < d2, both even (so A stays odd if a is odd)
    # B: (e1, e2) with e1 < e2, both even (so B stays even if b is even)
    
    # Small even differences: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    even_diffs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    
    # Try all pairs of difference sets
    for d1 in even_diffs:
        if time.time() - t_start > DEADLINE:
            break
        for d2 in even_diffs:
            if d2 <= d1:
                continue
            if time.time() - t_start > DEADLINE:
                break
            a_diffs = {d1, d2, d1 + d2}
            
            for e1 in even_diffs:
                if time.time() - t_start > DEADLINE:
                    break
                for e2 in even_diffs:
                    if e2 <= e1:
                        continue
                    if time.time() - t_start > DEADLINE:
                        break
                    b_diffs = {e1, e2, e1 + e2}
                    
                    if a_diffs & b_diffs:
                        continue
                    
                    # Now try small base values
                    # A = {a, a+d1, a+d1+d2}, a odd, 1 <= a, a+d1+d2 <= 128
                    # B = {b, b+e1, b+e1+e2}, b even, 1 <= b, b+e1+e2 <= 128
                    max_a = 128 - (d1 + d2)
                    max_b = 128 - (e1 + e2)
                    
                    # Try small a and b values
                    for a in range(1, min(max_a, 20) + 1, 2):  # odd a
                        if time.time() - t_start > DEADLINE:
                            break
                        A = [a, a + d1, a + d1 + d2]
                        if A[2] > 128:
                            continue
                        
                        for b in range(2, min(max_b, 20) + 1, 2):  # even b
                            if time.time() - t_start > DEADLINE:
                                break
                            B = [b, b + e1, b + e1 + e2]
                            if B[2] > 128:
                                continue
                            
                            # Check all 9 sums are prime
                            all_prime = True
                            for x in A:
                                for y in B:
                                    s = x + y
                                    if s > MAX_SUM or not is_prime[s]:
                                        all_prime = False
                                        break
                                if not all_prime:
                                    break
                            
                            if all_prime:
                                best_A = A
                                best_B = B
                                best_score = 9
                                found = True
                                break
                        
                        if found:
                            break
                    
                    if found:
                        break
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    # If not found with all-prime, try pairwise coprime (not necessarily prime)
    # Fall back to broader search
    odds = list(range(1, 61, 2))  # 1,3,5,...,59
    evens = list(range(2, 61, 2))  # 2,4,6,...,60
    
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
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) != 9:
                continue
            
            if _pairwise_coprime(list(sums)):
                best_A = A
                best_B = B
                best_score = 9
                found = True
                break
        
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    # Fallback: return best found so far
    return [best_A, best_B]
