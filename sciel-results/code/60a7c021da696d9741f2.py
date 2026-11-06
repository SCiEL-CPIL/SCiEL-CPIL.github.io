import math
import time

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Known valid solution with cardinality 9 (maximum possible for 3x3 sumsets)
    # A = [18, 42, 50], B = [1, 11, 17]
    # Sums: 19, 29, 35, 43, 53, 59, 51, 61, 67
    # All 9 distinct, all pairwise coprime
    known_A = [18, 42, 50]
    known_B = [1, 11, 17]
    
    # Verify the known solution
    def is_pairwise_coprime(nums):
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return False
        return True
    
    sums_known = [a + b for a in known_A for b in known_B]
    distinct_known = list(set(sums_known))
    
    if len(distinct_known) == 9 and is_pairwise_coprime(distinct_known):
        return [known_A, known_B]
    
    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    best_result = [known_A, known_B]
    best_score = len(distinct_known)
    
    # Strategy: Search for A, B where all 9 sums are distinct and pairwise coprime
    # Focus on finding solutions where sums are all prime (stronger condition)
    # or at least pairwise coprime composites
    
    # Phase 1: Systematic search over small values looking for all-prime sumsets
    small_range = list(range(1, 40))
    found_optimal = False
    
    for a1 in small_range:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in small_range:
            if a2 <= a1:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in small_range:
                if a3 <= a2:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                A = [a1, a2, a3]
                
                for b1 in small_range:
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                    for b2 in small_range:
                        if b2 <= b1:
                            continue
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b3 in small_range:
                            if b3 <= b2:
                                continue
                            if time.time() - t_start > TIME_LIMIT - 5:
                                break
                            B = [b1, b2, b3]
                            
                            # Compute all 9 sums
                            sums = [a + b for a in A for b in B]
                            
                            # Check if all are distinct
                            if len(set(sums)) < 9:
                                continue
                            
                            # Check if all are prime (stronger than pairwise coprime)
                            if all(is_prime[s] for s in sums):
                                best_result = [A, B]
                                best_score = 9
                                found_optimal = True
                                break
                        if found_optimal:
                            break
                    if found_optimal:
                        break
                if found_optimal:
                    break
            if found_optimal:
                break
        if found_optimal:
            break
    
    if found_optimal:
        return best_result
    
    # Phase 2: Broader search for pairwise coprime sums (not necessarily all prime)
    # Use a wider range but with time limits
    search_range = range(1, 60)
    
    for a1 in search_range:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in search_range:
            if a2 <= a1:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in search_range:
                if a3 <= a2:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                A = [a1, a2, a3]
                
                for b1 in search_range:
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                    for b2 in search_range:
                        if b2 <= b1:
                            continue
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b3 in search_range:
                            if b3 <= b2:
                                continue
                            if time.time() - t_start > TIME_LIMIT - 5:
                                break
                            B = [b1, b2, b3]
                            
                            sums = [a + b for a in A for b in B]
                            distinct_sums = list(set(sums))
                            card = len(distinct_sums)
                            
                            if card > best_score and is_pairwise_coprime(distinct_sums):
                                best_score = card
                                best_result = [A, B]
                                if card == 9:
                                    return best_result
    
    return best_result
