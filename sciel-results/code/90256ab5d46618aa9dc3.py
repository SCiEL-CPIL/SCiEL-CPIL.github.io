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

def _is_coprime_with_all(val, existing_set):
    """Check if val is coprime with all values in existing_set."""
    for v in existing_set:
        if _gcd(val, v) != 1:
            return False
    return True

def _search():
    """Search for optimal A, B using structured grid approach with composite coprime values."""
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
    
    # Also generate prime powers (p^k) that are <= 256 and pairwise coprime with other prime powers
    # A number is "useful" if it's a prime power (p^k for k>=1)
    # Two prime powers are coprime iff they are powers of different primes
    prime_powers = []
    for p in primes:
        val = p
        while val <= 256:
            prime_powers.append(val)
            val *= p
    
    # Build a set of "candidate sum values" that are prime powers
    # Two such values are coprime iff they're powers of different primes
    # So we need to pick 9 prime powers from distinct primes
    prime_power_set = set(prime_powers)
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Strategy 1: Grid approach where all 9 sums are prime powers of distinct primes
    # B = [b1, b2, b3], A = [a1, a2, a3]
    # Sums form a 3x3 grid: G[i][j] = A[i] + B[j]
    # All 9 must be distinct prime powers of distinct primes
    
    # For each pair of differences (d1, d2), find valid A values
    # Pre-compute for each b1, b2, b3 which a values give prime power sums
    
    # To be more efficient, iterate over b1 (small), d1, d2
    for b1 in range(1, 30):  # Keep b1 small to allow more room
        if time.time() > deadline:
            break
        for d1 in range(1, 50):
            if time.time() > deadline:
                break
            b2 = b1 + d1
            if b2 > 128:
                continue
            for d2 in range(1, 50):
                if time.time() > deadline:
                    break
                b3 = b2 + d2
                if b3 > 128:
                    continue
                
                B = [b1, b2, b3]
                
                # Find all a in [1, 128] such that a+b1, a+b2, a+b3 are all prime powers
                valid_as = []
                for a in range(1, 129):
                    s1 = a + b1
                    s2 = a + b2
                    s3 = a + b3
                    if s1 in prime_power_set and s2 in prime_power_set and s3 in prime_power_set:
                        valid_as.append(a)
                
                if len(valid_as) >= 3:
                    # Try combinations of 3 a's
                    # We need all 9 sums to be distinct and pairwise coprime
                    # Since they're prime powers, pairwise coprime means distinct primes
                    for combo in combinations(valid_as, 3):
                        A = list(combo)
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a + b)
                        
                        if len(sums) == 9:
                            # Check pairwise coprime (should be true if all are prime powers of distinct primes)
                            if _pairwise_coprime(list(sums)):
                                score = 9
                                if score > best_score:
                                    best_score = score
                                    best_A = A[:]
                                    best_B = B[:]
                                    if best_score >= 9:
                                        return best_A, best_B
                                break  # Only need to check one valid combo per (b1,d1,d2)
    
    # Strategy 2: Random search with bias toward coprime-friendly values
    rng = random.Random(42)
    # Create a pool of "good" numbers: primes and prime powers
    good_numbers = [n for n in range(1, 129) if n in prime_power_set or _is_prime(n)]
    if not good_numbers:
        good_numbers = list(range(1, 129))
    
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        # Mix: sometimes use good numbers, sometimes random
        if rng.random() < 0.7:
            A = list(rng.sample(good_numbers, 3))
            B = list(rng.sample(good_numbers, 3))
        else:
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
