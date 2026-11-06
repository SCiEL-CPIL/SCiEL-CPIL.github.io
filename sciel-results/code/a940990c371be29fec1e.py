import itertools
import math
import time

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
    # Max sum is 128 + 128 = 256
    max_val = 256
    
    # Sieve of Eratosthenes for quick prime check
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_pairwise_coprime(values):
        vals = list(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # Strategy: Find A, B such that all 9 sums A[i]+B[j] are distinct primes.
    # This guarantees pairwise coprime and maximum cardinality (9).
    
    # Precompute all pairs (a, b) where a+b is prime
    # We want to find a 3x3 grid of sums that are all distinct primes.
    
    # Let's iterate over possible A sets and B sets.
    # To speed up, we can fix A and search for B, or use a more clever search.
    
    # Since 128 choose 3 is ~340k, and for each we need to check B, it might be slow.
    # Let's try a randomized approach first, then systematic if needed.
    
    import random
    
    # Random search
    # Generate random A, then try to find B that makes all sums primes.
    
    # Actually, let's use a smarter approach.
    # We know that if A = {a1, a2, a3} and B = {b1, b2, b3},
    # the sums are a_i + b_j.
    
    # Let's try to construct such sets.
    # One known construction:
    # A = {1, 2, 3}, B = {2, 4, 6} -> sums: 3,5,7, 4,6,8, 5,7,9 -> not all prime/coprime
    # Let's search.
    
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # Seed result
    
    # Systematic search over small ranges first
    # Since the values are in [1, 128], let's try small values first.
    
    # We can iterate over all A = (a1, a2, a3) with 1 <= a1 < a2 < a3 <= 128
    # And for each, try all B = (b1, b2, b3) with 1 <= b1 < b2 < b3 <= 128
    # But that's 128^6 which is too large.
    
    # Better: iterate over A, and for each A, find if there exists a B.
    # For a fixed A, the sums are a1+b, a2+b, a3+b for each b in B.
    # We need all 9 sums to be pairwise coprime.
    
    # Let's try a random search with a time limit.
    
    rng = random.Random(42)
    
    def score_result(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if not is_pairwise_coprime(distinct_sums):
            return 0
        return len(distinct_sums)
    
    # Try the seed first
    seed_score = score_result([1, 7, 13], [4, 10, 16])
    if seed_score > best_score:
        best_score = seed_score
        best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Random search
    while time.time() - t_start < TIME_LIMIT:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = score_result(A, B)
        if s > best_score:
            best_score = s
            best_result = [A, B]
            if best_score == 9:
                break
    
    # If we haven't found 9, try a more systematic search
    if best_score < 9:
        # Try small values
        for a1 in range(1, 20):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a2 in range(a1+1, 20):
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                for a3 in range(a2+1, 20):
                    A = [a1, a2, a3]
                    for b1 in range(1, 20):
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b2 in range(b1+1, 20):
                            if time.time() - t_start > TIME_LIMIT - 5:
                                break
                            for b3 in range(b2+1, 20):
                                B = [b1, b2, b3]
                                s = score_result(A, B)
                                if s > best_score:
                                    best_score = s
                                    best_result = [A, B]
                                    if best_score == 9:
                                        break
                            if best_score == 9:
                                break
                        if best_score == 9:
                            break
                    if best_score == 9:
                        break
                if best_score == 9:
                    break
            if best_score == 9:
                break
    
    return [best_result[0], best_result[1]]
