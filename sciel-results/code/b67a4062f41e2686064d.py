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
    """Search for optimal A, B using structured grid approach with expanded ranges."""
    deadline = time.time() + 58  # 58s budget
    
    # Generate all primes in [2, 256]
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
            
    primes = [i for i in range(2, 257) if is_prime[i]]
    prime_set = set(primes)
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    def try_solution(A, B):
        nonlocal best_score, best_A, best_B
        if not _is_valid(A, B):
            return False
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) > best_score and _pairwise_coprime(list(sums)):
            best_score = len(sums)
            best_A = A[:]
            best_B = B[:]
            return best_score >= 9
        return False
    
    # Strategy 1: Structured search over difference pairs (d1, d2)
    # B = [1, 1+d1, 1+d1+d2], find A = [a1, a2, a3] such that all 9 sums are prime
    for d1 in range(1, 60):
        if time.time() > deadline:
            break
        for d2 in range(1, 60):
            if time.time() > deadline:
                break
            
            b1 = 1
            b2 = 1 + d1
            b3 = 1 + d1 + d2
            
            if b3 > 128:
                continue
            
            B = [b1, b2, b3]
            
            # Find valid a values where a+b1, a+b2, a+b3 are all prime
            valid_as = []
            for a in range(1, 129):
                if (a + b1) in prime_set and (a + b2) in prime_set and (a + b3) in prime_set:
                    valid_as.append(a)
                    
            if len(valid_as) >= 3:
                # Try to pick 3 a's that avoid collisions
                # Collision occurs if a_i - a_j = d1, d2, or d1+d2
                for combo in combinations(valid_as, 3):
                    A = list(combo)
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a + b)
                    if len(sums) == 9 and _pairwise_coprime(list(sums)):
                        if try_solution(A, B):
                            return best_A, best_B
                    # Early exit if we've found 9
                    if best_score >= 9:
                        return best_A, best_B
    
    # Strategy 2: Also try B starting from different base values
    for b_base in [1, 2, 3, 5, 7, 10]:
        if time.time() > deadline:
            break
        for d1 in range(1, 40):
            if time.time() > deadline:
                break
            for d2 in range(1, 40):
                if time.time() > deadline:
                    break
                
                b1 = b_base
                b2 = b_base + d1
                b3 = b_base + d1 + d2
                
                if b3 > 128:
                    continue
                
                B = [b1, b2, b3]
                
                valid_as = []
                for a in range(1, 129):
                    if (a + b1) in prime_set and (a + b2) in prime_set and (a + b3) in prime_set:
                        valid_as.append(a)
                        
                if len(valid_as) >= 3:
                    for combo in combinations(valid_as, 3):
                        A = list(combo)
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a + b)
                        if len(sums) == 9 and _pairwise_coprime(list(sums)):
                            if try_solution(A, B):
                                return best_A, best_B
                        if best_score >= 9:
                            return best_A, best_B
    
    # Strategy 3: Random search with smarter initialization
    rng = random.Random(12345)
    for _ in range(50000):
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
