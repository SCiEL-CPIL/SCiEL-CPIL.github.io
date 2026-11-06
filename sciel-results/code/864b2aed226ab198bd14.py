"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute primality for numbers up to 256 (128+128)
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct_vals = list(set(vals))
        n = len(distinct_vals)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(distinct_vals[i], distinct_vals[j]) > 1:
                    return False
        return True
    
    def score_of(A, B):
        """Compute the number of distinct pairwise-coprime sums."""
        sums = [a+b for a in A for b in B]
        distinct_sums = list(set(sums))
        if is_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Graph-based search for K3,3 subgraph where all edges are prime sums
    # Precompute prime sums table for 1..128
    MAX_VAL = 128
    prime_sum = [[False]*(MAX_VAL+1) for _ in range(MAX_VAL+1)]
    for a in range(1, MAX_VAL+1):
        for b in range(1, MAX_VAL+1):
            s = a + b
            if s <= MAX_SUM and is_prime[s]:
                prime_sum[a][b] = True

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Heuristic: Try to find 3 a's and 3 b's such that all 9 sums are prime
    # This ensures distinctness (primes are distinct if sums are distinct, 
    # and if a_i+b_j = a_k+b_l, then a_i-a_k = b_l-b_j. 
    # With 3x3 grid, collisions are possible but rare for primes.
    # Actually, if all 9 are prime, are they distinct?
    # 1+4=5, 2+3=5. Collision.
    # So we must check distinctness.
    
    # Sort candidates by "connectivity" to find rich nodes first
    a_candidates = list(range(1, MAX_VAL+1))
    b_candidates = list(range(1, MAX_VAL+1))
    
    # Sort by number of prime partners
    a_candidates.sort(key=lambda x: sum(1 for y in b_candidates if prime_sum[x][y]), reverse=True)
    b_candidates.sort(key=lambda x: sum(1 for y in a_candidates if prime_sum[y][x]), reverse=True)
    
    # Limit search to top 50 candidates to keep it fast
    top_a = a_candidates[:50]
    top_b = b_candidates[:50]
    
    for A in itertools.combinations(top_a, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        # Check if A has any common prime partner in B?
        # For a faster check, we can just iterate B
        for B in itertools.combinations(top_b, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            # Check if all 9 sums are prime
            all_prime = True
            sums_set = set()
            for a in A:
                for b in B:
                    if not prime_sum[a][b]:
                        all_prime = False
                        break
                    sums_set.add(a+b)
                if not all_prime:
                    break
            
            if all_prime:
                # Check distinctness
                if len(sums_set) == 9:
                    # Check pairwise coprime (implied by distinct primes > 1, but 2 is prime)
                    # Primes are pairwise coprime.
                    return [A, B]
                    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
