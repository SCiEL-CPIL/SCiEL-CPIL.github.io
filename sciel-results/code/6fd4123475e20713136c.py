"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools

# EVOLVE-BLOCK-START

# Precompute primes up to 256 (max sum 128+128)
MAX_SUM = 256
_is_prime = [False] * (MAX_SUM + 1)
_is_prime[2] = True
for _i in range(3, MAX_SUM + 1, 2):
    _is_prime[_i] = True
for _i in range(3, int(MAX_SUM**0.5) + 1, 2):
    if _is_prime[_i]:
        for _j in range(_i * _i, MAX_SUM + 1, _i * 2):
            _is_prime[_j] = False

def _pairwise_coprime(vals):
    """Check if all values in vals are pairwise coprime."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _find_best_solution():
    """Search for A, B with maximum cardinality of distinct pairwise-coprime sums."""
    t_start = time.time()
    TIME_LIMIT = 80  # Leave 10s safety margin
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Check the seed solution
    sums_seed = [a + b for a in best_A for b in best_B]
    distinct_seed = list(set(sums_seed))
    if _pairwise_coprime(distinct_seed):
        best_score = len(distinct_seed)
        
    # Target: 9 distinct pairwise coprime sums.
    # Strategy: Find A, B such that all 9 sums are distinct primes.
    # This guarantees pairwise coprimality.
    
    # Phase 1: Optimized search for all-prime distinct sums
    # A: odd, B: even
    # For a fixed A, valid b must satisfy: a+b is prime for all a in A.
    
    odd_candidates = list(range(1, 129, 2))
    even_candidates = list(range(2, 129, 2))
    
    # We only need to search a subset of A if we are confident a solution exists.
    # Given the density of primes, small A should work.
    
    for limit_a in [20, 40, 60, 128]:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
            
        # Use combinations of small odds first
        small_odds = [x for x in odd_candidates if x <= limit_a]
        
        for A in itertools.combinations(small_odds, 3):
            if time.time() - t_start > TIME_LIMIT - 2:
                break
                
            # Find valid B candidates
            # b is valid if a+b is prime for all a in A
            valid_bs = []
            for b in even_candidates:
                # Early exit if any a+b is not prime
                is_valid = True
                for a in A:
                    s = a + b
                    if s > MAX_SUM or not _is_prime[s]:
                        is_valid = False
                        break
                if is_valid:
                    valid_bs.append(b)
            
            if len(valid_bs) < 3:
                continue
                
            # Try combinations of valid_bs
            # We also need distinct sums: a_i + b_j distinct.
            # To speed this up, we can check a few random or first few combinations.
            # Since we want to find *any* valid B, let's try the first few.
            # Or better, iterate through combinations but break early if found.
            
            # Optimization: If we just pick 3 from valid_bs, do we guarantee distinct sums?
            # Not necessarily. But probability is high.
            # Let's iterate through combinations of valid_bs (limited count for speed)
            
            # To avoid slow loops, we can just pick the first 3 and check.
            # If they fail distinctness, try next 3, etc.
            # But iterating all C(k,3) might be slow if k is large.
            # Let's just pick a few specific ones or check distinctness during selection.
            
            # A simpler heuristic: Pick b1, b2, b3 such that they are far apart.
            # Or just iterate. valid_bs is usually small for small A.
            
            found = False
            # Limit the search for B to the first 20 valid candidates to keep it fast
            # If a solution exists, it's likely among small B values too.
            search_bs = valid_bs[:20]
            
            for B in itertools.combinations(search_bs, 3):
                sums = [a + b for a in A for b in B]
                if len(set(sums)) == 9:
                    return [list(A), list(B)]
            
            # If not found in first 20, try a wider range? 
            # For now, if limit_a is small, valid_bs is small.
            # If limit_a grows, valid_bs might grow, but search_bs is capped.
            
    # Phase 2: If no all-prime solution found, search for pairwise coprime composites
    # This is a fallback. The current code's phase 2 is O(N^6) which is too slow.
    # We can skip this if Phase 1 is robust, or implement a smarter fallback.
    # Given the density of primes, Phase 1 should almost always succeed for small A.
    
    return best_A, best_B

# Precompute the best solution at module load time
_t_precompute_start = time.time()
_precomputed_A, _precomputed_B = _find_best_solution()
_t_precompute_end = time.time()

# Fallback seed in case precomputation fails
_seed_A = [1, 7, 13]
_seed_B = [4, 10, 16]

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    # Return the precomputed solution
    if _precomputed_A and _precomputed_B:
        return [_precomputed_A, _precomputed_B]
    else:
        return [_seed_A, _seed_B]

# EVOLVE-BLOCK-END
