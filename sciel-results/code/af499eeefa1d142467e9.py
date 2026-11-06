"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for p in range(3, int(math.isqrt(n)) + 1, 2):
            if n % p == 0:
                return False
        return True

    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        if n % 2 == 0:
            factors.add(2)
            while n % 2 == 0:
                n //= 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
            p += 2
        if n > 1:
            factors.add(n)
        return factors

    # All odd values in [1, 128]
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))

    # Precompute prime factors for all possible sums (range 3 to 255, all odd)
    pf_cache = {}
    for n in range(3, 256, 2):
        pf_cache[n] = prime_factors(n)

    # Precompute compatibility: for each pair of odd sums (s1, s2), are they coprime?
    coprime_pair = {}
    odd_sums_range = list(range(3, 256, 2))
    for i in range(len(odd_sums_range)):
        for j in range(i, len(odd_sums_range)):
            s1, s2 = odd_sums_range[i], odd_sums_range[j]
            coprime_pair[(s1, s2)] = (s1 == s2) or (math.gcd(s1, s2) == 1)

    def are_coprime(s1, s2):
        if s1 == s2:
            return True
        if s1 > s2:
            s1, s2 = s2, s1
        return coprime_pair.get((s1, s2), math.gcd(s1, s2) == 1)

    def find_best_configuration():
        """Search for A, B with maximum distinct pairwise coprime sums."""
        
        # For a given A, find all valid b's where {a+b for a in A} are pairwise coprime
        def valid_bs_for_A(A):
            valid = []
            a1, a2, a3 = A
            for b in even_vals:
                s1 = a1 + b
                s2 = a2 + b
                s3 = a3 + b
                if not are_coprime(s1, s2):
                    continue
                if not are_coprime(s1, s3):
                    continue
                if not are_coprime(s2, s3):
                    continue
                valid.append(b)
            return valid
        
        def full_check(A, B):
            """Check if all 9 sums are distinct and pairwise coprime."""
            sums = []
            for a in A:
                for b in B:
                    sums.append(a + b)
            if len(set(sums)) != 9:
                return False
            for i in range(9):
                for j in range(i + 1, 9):
                    if math.gcd(sums[i], sums[j]) != 1:
                        return False
            return True
        
        best_A = None
        best_B = None
        
        # Generate candidate A's with better ordering:
        # Prefer A's with small values and moderate spacing
        candidate_As = []
        for a1 in range(1, 65, 2):
            for a2 in range(a1 + 2, 97, 2):
                for a3 in range(a2 + 2, 129, 2):
                    candidate_As.append((a1, a2, a3))
        
        # Phase 1: Systematic search with backtracking for B
        for idx, (a1, a2, a3) in enumerate(candidate_As):
            if time.time() > deadline - 15:
                break
            
            A = [a1, a2, a3]
            valid_b = valid_bs_for_A(A)
            
            if len(valid_b) < 3:
                continue
            
            # Backtracking: try to build B of size 3 from valid_b
            # Order valid_b by heuristic: prefer b's that give sums with rare primes
            found = False
            for b1, b2, b3 in combinations(valid_b[:25], 3):
                B = [b1, b2, b3]
                if full_check(A, B):
                    best_A = A[:]
                    best_B = B[:]
                    found = True
                    break
            
            if found:
                break
        
        if best_A is not None:
            return best_A, best_B
        
        # Phase 2: Random search with valid B pre-filtering
        import random
        random.seed(2024)
        
        for _ in range(300000):
            if time.time() > deadline - 10:
                break
            
            A = sorted(random.sample(odd_vals, 3))
            valid_b = valid_bs_for_A(A)
            if len(valid_b) >= 3:
                B = sorted(random.sample(valid_b, 3))
                if full_check(A, B):
                    best_A = A[:]
                    best_B = B[:]
                    break
        
        if best_A is not None:
            return best_A, best_B
        
        # Phase 3: Pure random without pre-filtering
        for _ in range(500000):
            if time.time() > deadline - 5:
                break
            A = sorted(random.sample(odd_vals, 3))
            B = sorted(random.sample(even_vals, 3))
            if full_check(A, B):
                best_A = A[:]
                best_B = B[:]
                break
        
        return best_A, best_B
    
    A, B = find_best_configuration()
    
    if A is not None and B is not None:
        return [A, B]
    else:
        # Fallback: known valid configuration with 5 distinct pairwise coprime sums
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
