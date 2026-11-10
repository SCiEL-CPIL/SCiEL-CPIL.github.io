"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache
from itertools import product as iter_product
import bisect

TARGET = factorial(12)  # 479001600

# Prime factorization: 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
PRIMES = [2, 3, 5, 7, 11]
EXPS = [10, 5, 2, 1, 1]
TARGET_EXP = tuple(EXPS)


def _build_divisors_with_exps(max_val):
    """Generate all divisors of TARGET that are > 12 and <= max_val, with their exponent tuples."""
    divs = []
    n = len(PRIMES)

    def gen(idx, cur, exps):
        if cur > max_val:
            return
        if idx == n:
            if cur > 12:
                divs.append((cur, tuple(exps)))
            return
        p = PRIMES[idx]
        for e in range(EXPS[idx] + 1):
            exps.append(e)
            gen(idx + 1, cur * (p ** e), exps)
            exps.pop()

    gen(0, 1, [])
    divs.sort(key=lambda x: x[0])
    return divs


def _search_partition_exp(divs_exp, target_exp, max_factors, deadline):
    """
    Find a subset of distinct divisors (with exponent vectors) whose exponent sum equals target_exp,
    using at most max_factors elements. Returns the list of values or None.
    Uses DFS with memoization on (idx, remaining_exp, rem_factors).
    """
    n = len(divs_exp)
    memo = {}
    
    # Precompute suffix products of remaining exponents for pruning?
    # Or just simple memoization.
    
    def dfs(idx, rem_exp, rem):
        if time.time() > deadline:
            return None
        if all(x == 0 for x in rem_exp):
            return []
        if rem == 0 or idx >= n:
            return None
        
        # Pruning: if the sum of all remaining exponents from idx onwards is less than rem_exp, fail
        # This is a bit expensive to compute on the fly, so we might skip it or precompute.
        # Let's just rely on memoization and the distinctness constraint.
        
        key = (idx, rem_exp, rem)
        if key in memo:
            return memo[key]
        
        # Try each divisor starting from idx
        for i in range(idx, n):
            val, exp = divs_exp[i]
            
            # Check if exp fits in rem_exp
            fits = True
            for j in range(len(exp)):
                if exp[j] > rem_exp[j]:
                    fits = False
                    break
            
            if not fits:
                continue
                
            new_rem_exp = tuple(rem_exp[j] - exp[j] for j in range(len(exp)))
            
            sub = dfs(i + 1, new_rem_exp, rem - 1)
            if sub is not None:
                memo[key] = [val] + sub
                return memo[key]
        
        memo[key] = None
        return None

    return dfs(0, target_exp, max_factors)


def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin before 90s kill

    # Get all divisors of TARGET > 12, sorted, with exponent vectors.
    all_divs_exp = _build_divisors_with_exps(TARGET)
    n_divs = len(all_divs_exp)
    
    # We want to minimize the largest factor.
    # The largest factor in any solution must be one of the divisors.
    # We iterate candidate bounds B (which are divisors > 12) in increasing order.
    # For a given bound B, we restrict our search to divisors <= B.
    # If we find a valid partition, B is a candidate for the maximum.
    # Since we iterate B in increasing order, the first valid partition found
    # will have the minimal possible maximum factor?
    # Wait. If we find a partition with max factor M, and we are currently testing bound B >= M,
    # the partition's max factor is actually max(partition).
    # If we iterate B in increasing order, and for each B we check if there is *any* partition
    # using divisors <= B, then the first B for which a partition exists is the minimal
    # possible *bound*. However, the actual max factor in that partition might be less than B
    # if we didn't pick the largest available divisor. But since we are checking feasibility
    # for bound B, if a solution exists with max factor < B, it would have been feasible
    # for a smaller bound.
    # So, iterating B in increasing order and returning the first feasible solution's
    # actual max factor should yield the optimal answer.
    # Actually, we can just return the partition found. The max of that partition is the answer.
    # And since we found it for the smallest B that allows *any* solution, and any solution
    # for bound B has max factor <= B, the max factor of the found solution is <= B.
    # Is it guaranteed to be minimal?
    # Suppose optimal max is M*. Then a solution exists with max factor M*.
    # This solution is feasible for any bound B >= M*.
    # It is NOT feasible for any bound B < M* (since all factors must be <= B < M*, so max factor < M*).
    # Therefore, the smallest B for which a solution exists is exactly M*.
    # And the solution we find for B=M* will have max factor <= M*. Since no solution exists
    # with max factor < M*, the max factor of our solution must be exactly M*.
    
    # So the strategy is:
    # 1. Iterate candidate bounds B from the smallest divisor > 12 upwards.
    # 2. For each B, check if there is a subset of distinct divisors (each <= B) whose product is TARGET.
    # 3. The first B for which this is true gives the minimal largest factor.
    # 4. Return the partition found.

    # Optimization: We can skip bounds that are too small.
    # The smallest factor in a partition must be > 12.
    # If we use k factors, the product is TARGET.
    # This doesn't give a tight lower bound on the max factor easily without solving the problem.
    # Just iterate.

    # We need to be careful with the DFS performance.
    # The number of divisors is ~782.
    # For small B, the list of available divisors is small, so DFS is fast.
    # For large B, the list is large, DFS might be slow.
    # But we stop as soon as we find a solution.
    # The optimal B is likely not the largest possible.
    
    # Let's precompute the list of divisors for each bound? No, just slice.
    # all_divs_exp is sorted by value.
    # For bound B = all_divs_exp[k][0], the available divisors are all_divs_exp[:k+1].
    
    for k in range(n_divs):
        if time.time() > deadline - 2.0:
            break
        
        bound = all_divs_exp[k][0]
        available = all_divs_exp[:k+1]
        
        # Quick check: is it even possible?
        # With at most 12 factors, max product is bound^12.
        # If bound^12 < TARGET, skip.
        # But bound is at least 13. 13^12 is huge.
        # Actually, we just need to know if a subset exists.
        
        # Run the search
        result = _search_partition_exp(available, TARGET_EXP, 12, deadline - 1.0)
        
        if result is not None:
            # Found a valid partition
            # Verify distinctness and > 12
            # The construction ensures distinctness (indices are increasing) and > 12 (filter in gen).
            # Verify product
            prod = 1
            for v in result:
                prod *= v
            if prod == TARGET and len(result) <= 12 and len(set(result)) == len(result) and all(v > 12 for v in result):
                return result
            else:
                # Should not happen, but fallback
                continue

    # Fallback: return [TARGET]
    return [TARGET]

# EVOLVE-BLOCK-END
