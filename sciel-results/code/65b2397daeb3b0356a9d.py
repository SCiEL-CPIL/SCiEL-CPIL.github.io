import sys
from math import factorial, log, isqrt
from functools import lru_cache
import time
import bisect

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Prime factorization: 2^10 * 3^5 * 5^2 * 7
    # Generate all divisors via prime factorization
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    # Number of divisors = 11 * 6 * 3 * 2 = 396

    def gen_divisors():
        divs = [1]
        for prime, exp in [(2, 10), (3, 5), (5, 2), (7, 1)]:
            new_divs = []
            p = 1
            for e in range(exp + 1):
                for d in divs:
                    new_divs.append(d * p)
                p *= prime
            divs = new_divs
        return divs

    all_divs = gen_divisors()
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)

    if n_divs == 0:
        return [target]

    # Precompute: for each divisor > 12, is it prime?
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

    # Precompute prime status for all divisors
    prime_status = {}
    for d in all_divs:
        prime_status[d] = is_prime(d)

    # For each divisor, count how many divisors of target it has (to help with pruning)
    # Actually, let's precompute the number of divisors of each divisor of target
    # A divisor d of target has num_div(d) divisors, where num_div counts divisors of d

    def count_divisors(n):
        """Count number of divisors of n using its factorization as a divisor of target."""
        # n divides target = 2^10 * 3^5 * 5^2 * 7
        # Find exponents
        e2 = 0
        tmp = n
        while tmp % 2 == 0:
            e2 += 1
            tmp //= 2
        e3 = 0
        while tmp % 3 == 0:
            e3 += 1
            tmp //= 3
        e5 = 0
        while tmp % 5 == 0:
            e5 += 1
            tmp //= 5
        e7 = 0
        while tmp % 7 == 0:
            e7 += 1
            tmp //= 7
        return (e2 + 1) * (e3 + 1) * (e5 + 1) * (e7 + 1)

    # Precompute divisor counts for all valid divisors
    div_counts = {}
    for d in valid_divs:
        div_counts[d] = count_divisors(d)

    # Check if we can decompose target into distinct divisors > 12, all <= max_val
    def check_solution(max_val):
        """
        Returns list of factors if found, else None.
        Uses DFS with aggressive pruning.
        """
        idx = bisect.bisect_right(valid_divs, max_val)
        candidates = valid_divs[:idx]
        n = len(candidates)

        if n == 0:
            return None

        # Quick check: product of all candidates must be >= target (actually must equal or we need subset)
        # Actually we need a SUBSET whose product is exactly target.
        # So the product of all candidates must be >= target, and we need to find a subset.

        # Better pruning: the product of the k largest candidates must be >= target for some k <= 12
        # And the product of the smallest k candidates must be <= target
        
        # Key insight: since all factors > 12, max number of factors is floor(log(12!)/log(13)) ≈ 7
        # 13^7 = 62748517, 13^8 = 815730721 > 479001600, so at most 7 factors

        # Let's use a different approach: meet-in-the-middle or smarter DFS
        
        # DFS with memoization on (index, remaining_product)
        # State space: index up to n, remaining is a divisor of target (at most 396 values)
        
        # Additional pruning:
        # 1. If rem is prime, it must be in candidates[i:]
        # 2. If rem has a prime factor > max_val, fail
        # 3. Lower bound: product of smallest remaining factors
        # 4. Upper bound: product of largest remaining factors
        
        # Precompute prefix/suffix products for pruning
        # Actually, for subset product, we need more careful pruning
        
        # Let me think differently. The key constraint is that we need DISTINCT factors.
        # Since factors are divisors of target and > 12, and we need their product to be target,
        # each prime's total exponent across all chosen factors must match target's exponents.
        
        # Alternative: work with exponent vectors.
        # target = 2^10 * 3^5 * 5^2 * 7^1
        # Each divisor d has an exponent vector (e2, e3, e5, e7)
        # We need to select distinct divisors whose exponent vectors sum to (10, 5, 2, 1)
        
        # This is a multi-dimensional subset sum problem!
        
        # Let's convert each divisor to its exponent vector
        def to_exp_vec(d):
            e2 = 0
            tmp = d
            while tmp % 2 == 0:
                e2 += 1
                tmp //= 2
            e3 = 0
            while tmp % 3 == 0:
                e3 += 1
                tmp //= 3
            e5 = 0
            while tmp % 5 == 0:
                e5 += 1
                tmp //= 5
            e7 = 0
            while tmp % 7 == 0:
                e7 += 1
                tmp //= 7
            return (e2, e3, e5, e7)

        target_vec = (10, 5, 2, 1)
        
        # Convert candidates to exponent vectors
        cand_vecs = [to_exp_vec(c) for c in candidates]
        
        # Now solve: select subset of cand_vecs (each used at most once) 
        # such that sum = target_vec, and |subset| <= 12
        
        # Use DFS with memoization on (index, remaining_vec)
        # remaining_vec is a tuple (r2, r3, r5, r7) where each >= 0
        
        # State space: index up to n, and remaining_vec components bounded by (10, 5, 2, 1)
        # So at most n * 11 * 6 * 3 * 2 = n * 396 states
        
        @lru_cache(maxsize=None)
        def dfs(i, r2, r3, r5, r7):
            if r2 == 0 and r3 == 0 and r5 == 0 and r7 == 0:
                return True
            if i >= n:
                return False
            
            # Pruning: if remaining exponents are all 0, done (handled above)
            # Pruning: check if we can possibly achieve the remaining exponents
            # with the remaining candidates
            # Sum of max possible exponents from candidates[i:]
            
            # Option 1: Skip candidates[i]
            if dfs(i + 1, r2, r3, r5, r7):
                return True
            
            # Option 2: Take candidates[i] if it fits
            c2, c3, c5, c7 = cand_vecs[i]
            if c2 <= r2 and c3 <= r3 and c5 <= r5 and c7 <= r7:
                if dfs(i + 1, r2 - c2, r3 - c3, r5 - c5, r7 - c7):
                    return True
            
            return False

        # Check if solution exists
        if not dfs(0, 10, 5, 2, 1):
            return None
        
        # Reconstruct solution
        result = []
        i, r2, r3, r5, r7 = 0, 10, 5, 2, 1
        while not (r2 == 0 and r3 == 0 and r5 == 0 and r7 == 0):
            if i >= n:
                break
            # Did we skip or take?
            if dfs(i + 1, r2, r3, r5, r7):
                i += 1
            else:
                # Must have taken candidates[i]
                c2, c3, c5, c7 = cand_vecs[i]
                result.append(candidates[i])
                r2 -= c2
                r3 -= c3
                r5 -= c5
                r7 -= c7
                i += 1
        
        return result

    # Binary search for the minimal maximum factor
    low = 0
    high = n_divs - 1
    best_result = None

    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        max_val = valid_divs[mid]
        
        sol = check_solution(max_val)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1

    if best_result is not None:
        return best_result
    
    # Fallback: trivial solution
    return [target]

# EVOLVE-BLOCK-END
