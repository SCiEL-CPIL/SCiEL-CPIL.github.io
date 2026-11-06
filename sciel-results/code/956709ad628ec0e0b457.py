"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B using structured grid approach."""
    deadline = time.time() + 58  # 58s budget
    
    # Generate all primes in [2, 256]
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
            
    primes = [i for i in range(2, 257) if is_prime[i]]
    
    # We look for a set of 9 pairwise coprime numbers (primes)
    # that can be arranged in a 3x3 grid with constant column differences.
    # Let the grid be G[i][j] = A[i] + B[j].
    # G[i][j+1] - G[i][j] = B[j+1] - B[j] = d_j (constant for all i).
    
    # Strategy:
    # 1. Iterate over combinations of 9 primes from the list of primes.
    #    This is C(54, 9) which is ~ 3e10. Too many.
    #    Instead, iterate over possible "differences" d1, d2.
    #    Or iterate over possible rows.
    
    # Better Strategy:
    # Iterate over all pairs of differences (d1, d2) such that d1, d2 are small.
    # B = [b1, b2, b3]. Let b1=1 (shift A).
    # Then B = [1, 1+d1, 1+d1+d2].
    # A = [a1, a2, a3].
    # Sums:
    # a1+1, a1+1+d1, a1+1+d1+d2
    # a2+1, a2+1+d1, a2+1+d1+d2
    # a3+1, a3+1+d1, a3+1+d1+d2
    
    # We need all 9 sums to be distinct primes.
    # Let's iterate over d1, d2 in a reasonable range (e.g., 1 to 50).
    # And iterate over a1, a2, a3 in [1, 128].
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Pre-compute primes set for O(1) lookup
    prime_set = set(primes)
    
    # To ensure we find a good solution quickly, we can also just run the previous random search in parallel?
    # No, let's stick to the deterministic one.
    
    # If d1, d2 are fixed, we check if there exist a1, a2, a3 such that
    # all 9 sums are primes.
    # This is a system of 9 primality constraints.
    
    # Iterate d1, d2
    for d1 in range(1, 50):
        if time.time() > deadline:
            break
        for d2 in range(1, 50):
            if time.time() > deadline:
                break
            
            b1 = 1
            b2 = 1 + d1
            b3 = 1 + d1 + d2
            
            if b3 > 128:
                continue
                
            # Check if b1, b2, b3 are distinct (they are if d1, d2 > 0)
            
            # Now find a1, a2, a3
            # a1, a2, a3 in [1, 128]
            # Sums:
            # a1+b1, a1+b2, a1+b3
            # a2+b1, a2+b2, a2+b3
            # a3+b1, a3+b2, a3+b3
            
            # We need all 9 to be prime.
            # Let's iterate a1, a2, a3
            # To speed up, we can pre-filter a values that make a+b1, a+b2, a+b3 prime.
            
            valid_as = []
            for a in range(1, 129):
                if (a + b1) in prime_set and (a + b2) in prime_set and (a + b3) in prime_set:
                    valid_as.append(a)
                    
            if len(valid_as) >= 3:
                # Pick 3 distinct a's from valid_as
                # We just need any 3.
                # But we also need the 9 sums to be distinct.
                # Since b's are distinct, and a's are distinct, sums are distinct?
                # a1+b1 = a2+b2 => a1-a2 = b2-b1 = d1.
                # If we pick a's carefully, we can avoid collisions.
                # But if we just pick any 3, might collide?
                # If a1 != a2, a1+b1 != a2+b1.
                # a1+b1 = a2+b2 => a1-a2 = d1.
                # So if we pick a's such that no two differ by d1 or d2 or d1+d2, we are safe.
                # Or just check the final set.
                
                # Try first 3
                A = valid_as[:3]
                B = [b1, b2, b3]
                
                sums = set()
                for a in A:
                    for b in B:
                        sums.add(a + b)
                        
                if len(sums) == 9 and _pairwise_coprime(list(sums)):
                    # Check bounds
                    if all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
                        score = 9
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return best_A, best_B
                                
    # Fallback to random search if deterministic fails (should not happen)
    rng = random.Random(12345)
    for _ in range(10000):
        if time.time() > deadline:
            break
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return best_A, best_B
                
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
