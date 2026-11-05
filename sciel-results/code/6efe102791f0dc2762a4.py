"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    def is_prime(n):
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
    
    def all_distinct_primes(sums):
        """Check if all sums are distinct and prime."""
        if len(set(sums)) != len(sums):
            return False
        return all(is_prime(s) for s in sums)
    
    def pairwise_coprime(vals):
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        if len(distinct_sums) < 2:
            return 0
        if not pairwise_coprime(distinct_sums):
            return 0
        return len(distinct_sums)
    
    # Precompute primes up to 256 (max sum = 128+128)
    primes_up_to_256 = [n for n in range(2, 257) if is_prime(n)]
    prime_set = set(primes_up_to_256)
    
    # Strategy: Find A and B such that all 9 sums are distinct primes
    # This gives |A+B| = 9, the maximum possible
    
    # Iterate over candidate A sets
    # To speed up, we can be smarter: for a given A, we need B such that
    # all a_i + b_j are primes. This means for each b_j, b_j must be such that
    # a_1+b_j, a_2+b_j, a_3+b_j are all prime.
    
    # Let's try a targeted search
    # For each pair of A candidates, check if there exists B
    
    # First, let's try small structured searches
    # A common trick: use A with small differences
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Try to find 9 distinct prime sums
    # Iterate over A = {a1, a2, a3} with a1 < a2 < a3
    # For efficiency, limit to reasonable ranges
    
    # Key insight: if a1, a2, a3 are all odd, then for b_j to make all sums prime (odd primes > 2),
    # b_j must be even (odd+even=odd). Similarly if a's are even, b's must be odd.
    # Actually, one sum could be 2, but let's aim for all odd primes.
    
    # Let's search: A with all elements same parity, B with opposite parity
    # This ensures all sums are odd (except possibly 2)
    
    # Try A with all odd, B with all even (or vice versa)
    # Actually, let's just do a smart search
    
    # For each candidate A, determine valid B values
    # A value b is valid for A if a_i + b is prime for all i
    
    # Precompute for each possible b, which A-sets work
    # Actually, let's iterate over A and for each, find valid B elements
    
    # To keep it fast, limit A to a subset
    # Let's try A from [1, 64] and B from [1, 128]
    
    # More efficient: iterate over A, for each A compute the set of b values
    # such that all a_i + b are prime, then pick 3 distinct such b values
    
    # Let's limit A search to keep it tractable
    # 128 choose 3 is ~350k, for each we need to check ~128 b values * 3 gcd checks
    # That's ~350k * 128 * 3 = ~134M operations, might be too slow
    
    # Better: precompute for each b, the set of a values such that a+b is prime
    # Then for a given A, b is valid if all a in A are in valid_a_for_b
    
    # Let's precompute: for each b in 1..128, which a in 1..128 give a+b prime
    # valid_a_for_b[b] = set of a such that a+b is prime
    
    valid_a_for_b = {}
    for b in range(1, 129):
        valid_a_for_b[b] = set()
        for a in range(1, 129):
            if (a + b) in prime_set:
                valid_a_for_b[b].add(a)
    
    # Now for each A (3-element set), find b values where all of A is in valid_a_for_b[b]
    # Then pick 3 such b values
    
    # To speed up, iterate over A in a structured way
    # Let's try A where elements are close together or have specific properties
    
    # Actually, let's just iterate but with early termination and deadline check
    
    # Iterate over A sets, but be smart about ordering
    # Try A with small values first (more likely to have prime sums with larger b)
    
    # Let's generate A candidates in order of "promisingness"
    # A set is promising if the intersection of valid_b for each a is large
    
    # For each a, valid_b_for_a[a] = set of b such that a+b is prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set()
        for b in range(1, 129):
            if (a + b) in prime_set:
                valid_b_for_a[a].add(b)
    
    # For a given A = {a1, a2, a3}, valid B elements are:
    # valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
    # We need |valid_b| >= 3
    
    # Iterate over A sets, compute valid_b, if |valid_b| >= 3, pick 3 and verify
    
    # To make this tractable, let's limit the search
    # Try A where all elements are in a reasonable range
    
    # Let's try: iterate over a1 from 1 to 64, a2 from a1+1 to 96, a3 from a2+1 to 128
    # But that's still ~64*32*32/6 ≈ 11k sets, very fast
    
    # Actually, let's be more thorough but with deadline
    found_9 = False
    
    # Try structured search first
    for a1 in range(1, 65):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 97):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                A = (a1, a2, a3)
                valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
                if len(valid_b) >= 3:
                    # Find 3 distinct b values
                    b_list = sorted(valid_b)
                    # Try first 3
                    B = (b_list[0], b_list[1], b_list[2])
                    sums = [a + b for a in A for b in B]
                    if len(set(sums)) == 9 and all(s in prime_set for s in sums):
                        # All 9 sums are distinct primes
                        best_A = list(A)
                        best_B = list(B)
                        best_count = 9
                        found_9 = True
                        break
            if found_9:
                break
        if found_9:
            break
    
    if found_9:
        return [best_A, best_B]
    
    # If 9 not found, try to find 8 or 7
    # For 8: we need 8 distinct pairwise coprime values
    # This could be 8 primes, or a mix
    
    # Let's search more broadly
    # For each A, find B that maximizes distinct pairwise coprime sums
    
    # Try all A from a limited set
    # Let's try A with elements in [1, 50]
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Iterate over A sets
    for a1 in range(1, 51):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 76):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                A = (a1, a2, a3)
                valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                # Try different B combinations
                # First try first 3
                for b_start in range(min(3, len(b_list) - 2)):
                    B = (b_list[b_start], b_list[b_start + 1], b_list[b_start + 2])
                    sums = [a + b for a in A for b in B]
                    distinct_sums = set(sums)
                    if len(distinct_sums) > best_count and pairwise_coprime(distinct_sums):
                        best_count = len(distinct_sums)
                        best_A = list(A)
                        best_B = list(B)
                        if best_count >= 9:
                            return [best_A, best_B]
    
    if best_A is not None:
        return [best_A, best_B]
    
    # Fallback: use the seed solution
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
