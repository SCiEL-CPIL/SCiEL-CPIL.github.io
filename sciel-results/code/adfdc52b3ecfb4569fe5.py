"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    # 90s limit, use 85s as deadline to be safe
    deadline = start_time + 85.0

    def pairwise_coprime(vals):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Compute the number of distinct sums that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Initial best solution (the seed, known to be valid with score 5)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_pair(best_A, best_B) # Should be 5

    # Precompute primes up to 256 (max sum 128+128)
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_sum)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime[j] = False

    # Strategy: Find A, B such that all 9 sums are distinct primes.
    # This guarantees pairwise coprimality and max cardinality 9.
    # Parity constraint: One set must be all odd, the other all even (or vice versa)
    # to ensure sums are odd (and thus potentially prime, except 2).
    # Since min sum is 1+2=3 (if 1 odd, 2 even) or 1+1=2 (if 1 odd, 1 odd -> sum even).
    # If we want sums > 2, we need one odd, one even.
    # If we allow sum 2, we can have 1+1. But we need 9 distinct sums.
    # 2 is the only even prime. If one sum is 2, others must be odd primes.
    # If A={1,3,5}, B={1,3,5}, sums are even. Only 2 is prime. Bad.
    # So, for 9 distinct primes, we likely need one set odd, one even.
    
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))

    def search_case(A_source, B_source, A_name, B_name):
        nonlocal best_A, best_B, best_score
        
        # To keep it fast, we don't iterate all C(64,3)^2.
        # We iterate A, and for each A, find compatible B values.
        # Compatible b for A={a1,a2,a3} means a1+b, a2+b, a3+b are all prime.
        
        # Optimization: Precompute for each b in B_source, which a's in A_source make a+b prime?
        # No, just iterate A.
        
        # Limit search space to smaller numbers first to find a solution quickly.
        # If found, score is 9, which is max. Return immediately.
        
        # Phase 1: Small values (1-40)
        limit1 = 40
        A_cands = [x for x in A_source if x <= limit1]
        B_cands = [x for x in B_source if x <= limit1]
        
        for i1 in range(len(A_cands)):
            if time.time() > deadline: return
            a1 = A_cands[i1]
            for i2 in range(i1 + 1, len(A_cands)):
                a2 = A_cands[i2]
                for i3 in range(i2 + 1, len(A_cands)):
                    a3 = A_cands[i3]
                    A = [a1, a2, a3]
                    
                    # Find compatible b's
                    compat_b = []
                    for b in B_cands:
                        if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                            compat_b.append(b)
                            
                    if len(compat_b) >= 3:
                        # Try first 3
                        B = compat_b[:3]
                        s = score_pair(A, B)
                        if s > best_score:
                            best_score = s
                            best_A = A[:]
                            best_B = B[:]
                            if best_score == 9:
                                return
                        # Try other combos if time permits
                        if len(compat_b) > 3 and time.time() < deadline - 10:
                            for k in range(1, min(5, len(compat_b) - 2)):
                                B2 = compat_b[k:k+3]
                                s2 = score_pair(A, B2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_A = A[:]
                                    best_B = B2[:]
                                    if best_score == 9:
                                        return

    # Case 1: A odd, B even
    search_case(odd_vals, even_vals, "odd", "even")
    
    # Case 2: A even, B odd
    if time.time() < deadline - 5:
        search_case(even_vals, odd_vals, "even", "odd")

    return [best_A, best_B]

# EVOLVE-BLOCK-END
