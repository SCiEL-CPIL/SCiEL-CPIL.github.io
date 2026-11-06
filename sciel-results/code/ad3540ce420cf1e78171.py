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
    
    # Hardcoded known solution: A=[1, 5, 11], B=[2, 4, 8]
    # Sums: 1+2=3, 1+4=5, 1+8=9(no!) -- wait, 9 is not prime.
    # Let me verify a correct one.
    # Try A=[1, 7, 13], B=[2, 6, 10]:
    # 1+2=3, 1+6=7, 1+10=11
    # 7+2=9(no) -- 9 not prime.
    
    # Let me search for a verified solution programmatically first.
    # A=[1, 3, 5], B=[2, 4, 8]:
    # 1+2=3, 1+4=5, 1+8=9(no)
    
    # A=[1, 7, 11], B=[2, 4, 8]:
    # 1+2=3, 1+4=5, 1+8=9(no)
    
    # A=[3, 7, 11], B=[2, 4, 8]:
    # 3+2=5, 3+4=7, 3+8=11
    # 7+2=9(no)
    
    # A=[1, 3, 11], B=[2, 4, 8]:
    # 1+2=3, 1+4=5, 1+8=9(no)
    
    # A=[5, 7, 11], B=[2, 4, 8]:
    # 5+2=7, 5+4=9(no)
    
    # A=[1, 5, 13], B=[2, 4, 8]:
    # 1+2=3, 1+4=5, 1+8=9(no)
    
    # Let me just do a fast targeted search for small values.
    # We need all 9 sums to be distinct and pairwise coprime (ideally all prime).
    
    # Quick search: try small odd A and even B
    odds = list(range(1, 31, 2))   # 1,3,5,...,29
    evens = list(range(2, 31, 2))  # 2,4,6,...,30
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s == 9:
                    return [best_A, best_B]
    
    # If we didn't find 9 in the small range, expand
    odds_big = list(range(1, 61, 2))
    evens_big = list(range(2, 61, 2))
    
    for A in itertools.combinations(odds_big, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(evens_big, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
