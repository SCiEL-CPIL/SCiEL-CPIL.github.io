"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill

    # Precompute sieve for primality up to 256 (max sum = 128+128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def count_distinct_coprime(A, B):
        """Count distinct values in A+B that are pairwise coprime. Return 0 if not all are pairwise coprime."""
        sums = set(a + b for a in A for b in B)
        if is_pairwise_coprime(sums):
            return len(sums)
        return 0

    # Best known result from seed: 5 distinct coprime sums
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Strategy: Fix B to small prime sets and search for A
    # Key insight: If B = {p1, p2, p3} are small primes, then for A = {a1, a2, a3},
    # we need all 9 sums a_i + p_j to be distinct primes.
    
    # Candidate B sets: small primes and small integers
    candidate_B_sets = []
    
    # Try B as small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    for b1 in small_primes:
        for b2 in small_primes:
            if b2 <= b1:
                continue
            for b3 in small_primes:
                if b3 <= b2:
                    continue
                candidate_B_sets.append([b1, b2, b3])
    
    # Also try B with mixed parity (one even, two odd or vice versa)
    for b1 in range(1, 50):
        for b2 in range(b1 + 1, 50):
            for b3 in range(b2 + 1, 50):
                if b1 + 128 <= 256 and b3 + 128 <= 256:
                    candidate_B_sets.append([b1, b2, b3])
    
    # For each candidate B, search for valid A
    # Precompute: for each b, which a values make a+b prime
    prime_a_for_b = {}
    for b in range(1, 129):
        prime_a_for_b[b] = set()
        for a in range(1, 129):
            if is_prime[a + b]:
                prime_a_for_b[b].add(a)
    
    # Search over candidate B sets
    for B in candidate_B_sets:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        
        b1, b2, b3 = B
        
        # Find common A values that make all sums prime
        common_a = prime_a_for_b[b1] & prime_a_for_b[b2] & prime_a_for_b[b3]
        
        if len(common_a) < 3:
            continue
        
        a_list = sorted(common_a)
        
        # Try all triples of a from common_a
        for a1_idx, a1 in enumerate(a_list):
            for a2_idx, a2 in enumerate(a_list):
                if a2_idx <= a1_idx:
                    continue
                for a3_idx, a3 in enumerate(a_list):
                    if a3_idx <= a2_idx:
                        continue
                    
                    A = [a1, a2, a3]
                    
                    # Check all 9 sums are distinct primes
                    sums = set()
                    valid = True
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in sums or not is_prime[s]:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid and len(sums) == 9:
                        return [A, B]
    
    # Fallback: Original search approach
    # Precompute for each a in [1,128], the set of b in [1,128] where a+b is prime.
    prime_b_for_a = {}
    for a in range(1, 129):
        prime_b_for_a[a] = set()
        for b in range(1, 129):
            if is_prime[a + b]:
                prime_b_for_a[a].add(b)
    
    # Search Case 1: A even, B odd
    even_as = [a for a in range(2, 129, 2)]
    odd_bs = [b for b in range(1, 129, 2)]
    
    for a1_idx, a1 in enumerate(even_as):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2_idx, a2 in enumerate(even_as):
            if a2_idx <= a1_idx:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3_idx, a3 in enumerate(even_as):
                if a3_idx <= a2_idx:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                
                common_b = prime_b_for_a[a1] & prime_b_for_a[a2] & prime_b_for_a[a3]
                
                if len(common_b) < 3:
                    continue
                
                b_list = sorted(common_b)
                
                for b1_idx, b1 in enumerate(b_list):
                    for b2_idx, b2 in enumerate(b_list):
                        if b2_idx <= b1_idx:
                            continue
                        for b3_idx, b3 in enumerate(b_list):
                            if b3_idx <= b2_idx:
                                continue
                            
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums or not is_prime[s]:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                return [A, B]
    
    # Fallback: brute force with smaller range
    range2 = range(1, 40)
    for A in itertools.combinations(range2, 3):
        if time.time() - t_start > TIME_LIMIT - 3:
            break
        for B in itertools.combinations(range2, 3):
            score = count_distinct_coprime(list(A), list(B))
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                if best_score == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
