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

    def pairwise_coprime_check(sums_set):
        """Check if all elements in sums_set are pairwise coprime."""
        slist = list(sums_set)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if math.gcd(slist[i], slist[j]) != 1:
                    return False
        return True

    # All odd values in [1, 128]
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))

    # Precompute prime factors for all possible sums (range 3 to 255, all odd)
    pf_cache = {}
    for n in range(3, 256, 2):
        pf_cache[n] = prime_factors(n)

    # Precompute compatibility: for each pair of odd sums (s1, s2), are they coprime?
    # This is O(128^2) = ~16k checks, fast
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

    # Key insight: A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: s_ij = a_i + b_j
    # For fixed A, the 9 sums are determined by B.
    # s_1j - s_1k = b_j - b_k (same for all rows)
    # s_i1 - s_k1 = a_i - a_k (same for all cols)
    
    # Strategy: enumerate A, then for each A find B such that all 9 sums are pairwise coprime.
    # For a given A and candidate B, the 9 sums must all be pairwise coprime.
    
    # Better strategy: 
    # Fix A = {a1, a2, a3}. For B = {b1, b2, b3}, the sums are:
    # a1+b1, a1+b2, a1+b3
    # a2+b1, a2+b2, a2+b3
    # a3+b1, a3+b2, a3+b3
    # All 9 must be pairwise coprime (and distinct for max cardinality).
    
    # For efficiency, precompute for each (a, b) the prime factors of a+b.
    # Then check pairwise coprimality via factor sets.

    # Score-based approach:
    # For each odd a, compute which even b values give sums with "rare" prime factors
    # (i.e., primes that appear in few other sums).

    # Let's use a smarter enumeration:
    # The 9 sums must be pairwise coprime. This is a very strong constraint.
    # In particular, no two sums can share a prime factor.
    
    # For a given A, let's find valid B values.
    # A value b is "compatible" with A if all three sums a1+b, a2+b, a3+b are pairwise coprime
    # to each other AND to each other's prime factors.
    # Then we need 3 mutually compatible b values where all cross-sums are also coprime.

    def find_best_configuration():
        """Systematic search for A, B with 9 distinct pairwise coprime sums."""
        
        # Precompute: for each pair (a, b), the set of prime factors of a+b
        # and a quick coprime check structure
        
        # For each odd a and even b, compute pf of (a+b)
        # Store as a dict: sum_val -> set of primes
        
        # Strategy: iterate over A combinations, then greedily build B
        
        # To speed up, precompute for each a the list of (b, sum_val, pf_set) 
        # sorted by some heuristic
        
        # Actually, let's be more direct:
        # For a given A, a candidate b is valid if:
        # 1. All three sums a1+b, a2+b, a3+b are pairwise coprime
        # 2. None of these sums equals any other sum (for distinctness)
        # Then for B = {b1, b2, b3}, additionally:
        # 3. For all i≠j, a_i+b_j must be coprime to a_k+b_l for all (k,l)≠(i,j)
        
        # Let's precompute for each a, b: the sum and its prime factors
        sum_pfs = {}
        for a in odd_vals:
            for b in even_vals:
                s = a + b
                sum_pfs[(a, b)] = pf_cache[s]
        
        # For a given A, find all valid b's
        def valid_bs_for_A(A):
            """Find all b in even_vals such that {a+b for a in A} are pairwise coprime."""
            valid = []
            a1, a2, a3 = A
            for b in even_vals:
                s1 = a1 + b
                s2 = a2 + b
                s3 = a3 + b
                # Check pairwise coprime among these 3
                if not are_coprime(s1, s2):
                    continue
                if not are_coprime(s1, s3):
                    continue
                if not are_coprime(s2, s3):
                    continue
                valid.append(b)
            return valid
        
        # For a given A and candidate B, check full validity
        def full_check(A, B):
            """Check if all 9 sums are distinct and pairwise coprime."""
            sums = []
            for a in A:
                for b in B:
                    sums.append(a + b)
            # Check distinctness
            if len(set(sums)) != 9:
                return False
            # Check pairwise coprime
            for i in range(9):
                for j in range(i + 1, 9):
                    if math.gcd(sums[i], sums[j]) != 1:
                        return False
            return True
        
        # Search strategy:
        # 1. Try structured A values (small, well-separated)
        # 2. For each A, find valid B candidates
        # 3. Among valid B candidates, find 3 that work together
        
        # Precompute valid B sets for common A patterns
        # Use a greedy + backtracking approach
        
        best_A = None
        best_B = None
        
        # Generate candidate A's: prefer small values with good spacing
        # A = {a1, a2, a3} with a1 < a2 < a3
        # The differences d1 = a2-a1, d2 = a3-a2 affect the sum structure
        
        # Try A's from small values first
        candidate_As = []
        for a1 in range(1, 65, 2):
            for a2 in range(a1 + 2, 97, 2):
                for a3 in range(a2 + 2, 129, 2):
                    candidate_As.append((a1, a2, a3))
        
        # Sort by some heuristic: prefer A's where the differences are "nice"
        # Actually, just try in order but with a time limit
        
        # For speed, let's precompute valid B's for each A
        # But that's C(64,3) = ~41k A's, each requiring scanning 64 B's
        # That's ~2.6M checks, each with 3 gcd checks = ~8M gcd ops. Should be fast enough.
        
        # Let's do it more efficiently:
        # For each A, compute valid B's, then try to pick 3 compatible B's
        
        # Precompute: for each pair of b values (b1, b2) and A, 
        # check if the 6 sums {a_i+b_j} are pairwise coprime
        # This is complex, so let's just do the full check when we have 3 B candidates
        
        # Simpler approach: for each A, get valid B list, then try all C(|valid|, 3)
        # If valid list is short, this is fast
        
        # Let's limit: only try A's where the valid B list has at least 3 elements
        
        # Phase 1: Systematic search
        for idx, (a1, a2, a3) in enumerate(candidate_As):
            if time.time() > deadline - 15:
                break
            
            A = [a1, a2, a3]
            valid_b = valid_bs_for_A(A)
            
            if len(valid_b) < 3:
                continue
            
            # Try to find 3 compatible B values from valid_b
            # First, try the first few
            found = False
            for b1, b2, b3 in combinations(valid_b[:20], 3):
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
        
        # Phase 2: If systematic search failed, try random search
        import random
        random.seed(2024)
        
        for _ in range(200000):
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
        
        # Phase 3: Even more random, without pre-filtering
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
        # Fallback: return a known valid configuration
        # A = {1, 7, 13}, B = {4, 10, 16}
        # Sums: 5,11,17, 11,17,23, 17,23,29 -> distinct: 5,11,17,23,29
        # Check: gcd(5,11)=1, gcd(5,17)=1, gcd(5,23)=1, gcd(5,29)=1,
        #        gcd(11,17)=1, gcd(11,23)=1, gcd(11,29)=1,
        #        gcd(17,23)=1, gcd(17,29)=1, gcd(23,29)=1. All good!
        # Cardinality = 5
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
