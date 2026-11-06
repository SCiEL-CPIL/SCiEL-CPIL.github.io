import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    LIMIT = 10**18
    # Primes to consider. 18 primes is a good upper bound for square-free products near 10^18
    # 2*3*5*...*59 (16 primes) is approx 3.2e18, so 15 primes is safe for exp 1.
    # With exp 2, fewer primes fit.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_tau_perp = 0
    best_n = 2  # Fallback
    
    def calc_tau_perp_from_factors(factors):
        """
        Calculate tau_perp directly from factorization.
        factors: list of (p, exp)
        """
        if not factors:
            return 0
            
        # Generate divisors
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        
        divisors.sort()
        
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score

    def add_candidate(n, factors):
        nonlocal best_tau_perp, best_n
        if n > LIMIT or n < 2:
            return
        tau_perp = calc_tau_perp_from_factors(factors)
        if tau_perp > best_tau_perp:
            best_tau_perp = tau_perp
            best_n = n

    # Phase 1: Search for numbers with exponents 1 and 2 (and 0)
    # We use DFS to generate combinations of exponents for the first K primes
    # We limit K to 18 to keep the search space manageable with exps up to 2
    # If we use exps up to 3, K can be smaller
    
    def search_low_exponents(max_prime_idx, current_n, current_factors, max_exp_allowed):
        if time.time() > deadline:
            return
            
        # Evaluate current state
        if current_n > 1:
            add_candidate(current_n, current_factors)
            
        if max_prime_idx >= len(primes):
            return
            
        p = primes[max_prime_idx]
        
        # Determine max exponent for this prime
        # We want to try exponents from max_exp_allowed down to 1, and also 0 (skip)
        # But since we are building the number, "0" means we just recurse to the next prime without adding this one.
        
        # Calculate max possible exp based on LIMIT
        limit_for_p = LIMIT // current_n
        if limit_for_p < p:
            return
            
        max_possible_exp = 0
        temp = 1
        while temp * p <= limit_for_p and max_possible_exp < max_exp_allowed:
            temp *= p
            max_possible_exp += 1
            
        if max_possible_exp == 0:
            # Can't add this prime at all, but we can skip it and try next?
            # In this specific structure (consecutive primes), if we can't fit p, we can't fit larger primes either.
            # So we return.
            return

        # Option 1: Skip this prime (exponent 0)
        # This allows us to find combinations that don't include all first K primes, 
        # though typically for maximizing divisors/coprimality, including smaller primes is better.
        # However, if we include a prime, we must include all smaller ones? 
        # Not necessarily for this metric, but for "first k primes" structure, yes.
        # Let's assume we use a prefix of primes. If we skip p, we stop? 
        # Actually, if we skip p, we might still be able to use p+1? 
        # No, if n is composed of a subset of primes, the "adjacent divisors" property depends on the set.
        # Usually, the optimal set is the first k primes.
        # So if we skip p, we should probably stop adding primes, or only consider larger primes if we have room.
        # But mixing small and large primes is generally suboptimal for this metric compared to dense small primes.
        # So we assume the set of primes is {p_1, ..., p_k}.
        
        # Iterate exponents from max_possible_exp down to 1
        for exp in range(max_possible_exp, 0, -1):
            next_n = current_n * (p ** exp)
            next_factors = current_factors + [(p, exp)]
            
            # Recurse to next prime
            # Next max exp can be the same or lower? 
            # For "first k primes" structure, there is no strict non-increasing constraint on exponents 
            # unless we want to prune. But for low exponents (1,2), it doesn't matter much.
            # Let's allow up to max_exp_allowed for the next prime too, to be safe.
            search_low_exponents(max_prime_idx + 1, next_n, next_factors, max_exp_allowed)
            
            if time.time() > deadline:
                return
        
        # Option: Stop adding primes (we already evaluated current_n above)
        # We don't need to explicitly "skip" and continue to next prime with exp 0 
        # if we assume the set of primes is a prefix. 
        # If we allow non-prefix subsets, the search space explodes.
        # Given the nature of the problem, prefix subsets are the strongest candidates.

    # Start Phase 1: Exponents up to 2
    # We start with empty factorization
    search_low_exponents(0, 1, [], 2)
    
    # Phase 2: Refine with higher exponents for the smallest primes
    # We take the best candidate found so far and try to modify it
    # Actually, the DFS above already explores exp=1 and exp=2 for all primes in a prefix.
    # It does NOT explore exp=3, 4, etc.
    
    # Let's do a targeted search for exp=3 on the first few primes
    def search_high_exponents_small(max_prime_idx, current_n, current_factors, max_exp_allowed):
        if time.time() > deadline:
            return
            
        if current_n > 1:
            add_candidate(current_n, current_factors)
            
        if max_prime_idx >= len(primes):
            return
            
        p = primes[max_prime_idx]
        limit_for_p = LIMIT // current_n
        if limit_for_p < p:
            return
            
        max_possible_exp = 0
        temp = 1
        while temp * p <= limit_for_p and max_possible_exp < max_exp_allowed:
            temp *= p
            max_possible_exp += 1
            
        if max_possible_exp == 0:
            return

        for exp in range(max_possible_exp, 0, -1):
            next_n = current_n * (p ** exp)
            next_factors = current_factors + [(p, exp)]
            
            # For higher exponents, we likely have fewer primes.
            # So we can restrict the next max exp to be lower or same.
            search_high_exponents_small(max_prime_idx + 1, next_n, next_factors, max_exp_allowed)
            
            if time.time() > deadline:
                return

    # Run Phase 2: Exponents up to 5 for the first 10 primes
    # This covers cases like 2^5 * 3^4 * 5^3 ...
    search_high_exponents_small(0, 1, [], 5)
    
    # Phase 3: Local Search / Polish
    # If we have a good candidate, try swapping primes or adjusting exponents slightly?
    # The DFS above is quite comprehensive for the "prefix" structure.
    # Let's ensure we didn't miss the case where we skip a prime in the middle?
    # Unlikely to be optimal.
    
    if best_n == 2:
        # Fallback if something went wrong
        best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    return best_n

# To ensure we don't miss the "pure square-free" high prime count case, 
# the search_low_exponents with max_exp=2 covers exp=1.
# The search_high_exponents_small with max_exp=5 covers higher powers.

# One potential issue: The DFS `search_low_exponents` assumes we MUST use a prefix of primes.
# i.e., if we use prime 7, we must have used 2, 3, 5.
# This is a strong assumption. Is it always true?
# For maximizing the number of divisors, yes.
# For tau_perp, it is highly likely yes, because small primes create more "mixing" in the divisor list.
# Replacing a small prime with a large one reduces the number of divisors and likely the coprime transitions.

# Let's double check the time complexity.
# Phase 1: Branching factor ~3 (exp 0, 1, 2). Depth ~18. 3^18 is too big.
# But the product constraint prunes heavily.
# 2^1 * 3^1 * ... * p_k^1 <= 10^18.
# Primorial 47 (15 primes) is ~6e17. Primorial 53 (16 primes) is ~3e19.
# So max depth for exp=1 is 15.
# With exp=2, depth is smaller.
# The number of valid states is the number of "smooth" numbers <= 10^18 with exponents <= 2.
# This is manageable.

# Phase 2: Branching factor ~6 (exp 0..5). Depth ~10.
# 6^10 is ~60 million, but pruned by product.
# 2^5 * 3^5 * 5^5 ... is small.
# This should be fast enough.

# Final check on the code structure.
# The function `construct` is the entry point.
# It returns an integer.
# It uses standard library only.

# One optimization: `calc_tau_perp_from_factors` generates divisors and sorts them.
# For n with many divisors (e.g. 2^5 * 3^4 * 5^3 ...), the number of divisors can be in the thousands.
# Sorting and GCD is O(D log D). D is small enough.

# Let's refine the deadline check to be more frequent in the inner loops if needed, 
# but the current checks at the start of recursive calls should suffice.

# Re-verify the "prefix" assumption.
# Suppose n = 2 * 3 * 5 * 11. We skipped 7.
# Divisors: 1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330.
# If we used 7 instead of 11: n = 2 * 3 * 5 * 7 = 210.
# Divisors: 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210.
# Generally, smaller primes yield more divisors and better mixing.
# So the prefix assumption is safe.

# Final Code Block
