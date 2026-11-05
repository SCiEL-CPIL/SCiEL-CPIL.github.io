"""Combined program: exhaustive exact search to minimize the largest factor in a
distinct-factorization of 12! with all factors > 12."""
import time
from math import factorial
from itertools import combinations

_START = time.time()
_LIMIT = 80.0          # safety margin well under the 90 s kill
_N = 12
_TARGET = factorial(_N)   # 479001600 = 2^10 * 3^5 * 5^2 * 7


def _remaining_time():
    return _LIMIT - (time.time() - _START)


def _is_valid(sol):
    if not (1 <= len(sol) <= _N):
        return False
    if len(set(sol)) != len(sol):
        return False
    if any(x <= _N for x in sol):
        return False
    prod = 1
    for x in sol:
        prod *= x
        if prod > _TARGET:
            return False
    return prod == _TARGET


def _all_prime_factors(n):
    """Return sorted list of all divisors of n that are > 12, distinct, and product-compatible."""
    # Precompute all divisors of _TARGET that are > 12
    divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7
    # Enumerate all divisors by iterating exponents
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    d = (1 << e2) * (3 ** e3) * (5 ** e5) * (7 ** e7)
                    if d > _N:
                        divisors.append(d)
    return sorted(set(divisors))


def construct():
    best_solution = None
    best_largest = float('inf')

    def update(parts):
        nonlocal best_solution, best_largest
        if not _is_valid(parts):
            return
        mx = max(parts)
        if mx < best_largest:
            best_largest = mx
            best_solution = sorted(parts)

    # Precompute all valid divisors > 12
    valid_divs = _all_prime_factors(_TARGET)
    
    # Sort by value for potential pruning
    valid_divs.sort()

    # We need to select a subset of valid_divs (distinct, product == _TARGET)
    # such that all are > 12, count in 1..12, minimizing the max.
    #
    # Strategy: Branch and bound.
    # We build a list of factors in increasing order to avoid permutations.
    # At each step, we pick the next factor >= last picked factor.
    # We prune if:
    # 1. Product of current factors * (product of smallest possible remaining) > _TARGET? No, we need exact product.
    # 2. We track the remaining product needed.
    # 3. If the remaining product is 1, we are done (if we have at least 1 factor).
    # 4. If the remaining product is > 1, we must pick more factors.
    # 5. The next factor must be a divisor of the remaining product.
    # 6. To ensure distinctness and ordering, next factor > last factor.
    # 7. Pruning: if the remaining product is prime and > last_factor, we can take it.
    #    If remaining product is composite, we must split it.
    #
    # This is a subset product problem with constraints.
    # Given the small number of prime factors (10+5+2+1 = 18 total prime factors),
    # the number of distinct divisors is 11*6*3*2 = 396.
    # The search space is manageable with good pruning.

    # We'll use a recursive search.
    # State: (remaining_product, last_factor, current_factors)
    # To enforce distinctness and avoid permutations, we only pick factors > last_factor.
    # Also, the product of current_factors * remaining_product must equal _TARGET.
    
    # Optimization: Precompute for each divisor d, the list of divisors of _TARGET/d that are > d.
    # This is expensive to precompute for all d, so we compute on the fly or use a helper.

    # Better approach:
    # Since the number of factors is small (<= 12) and the values are bounded,
    # we can iterate over the number of factors k from 1 to 12.
    # For each k, try to find k distinct divisors > 12, product = _TARGET, minimizing max.
    # Actually, minimizing max is equivalent to finding a solution with the smallest possible max.
    # We can iterate candidate max values M from smallest to largest, and check if a solution exists with all factors <= M.
    # The smallest M for which a solution exists is the answer.
    
    # Let's try this approach:
    # Sort valid_divs.
    # For M in valid_divs:
    #   Check if there exists a subset of valid_divs[:index_of_M] (inclusive) with product _TARGET.
    #   This is a subset product problem, which is NP-complete in general, but here the numbers are small.
    #   We can use a DP or backtracking.
    #   Given the small number of divisors (396) and the fact that the product is fixed, backtracking with pruning should work.
    #   Pruning: 
    #     - Product of chosen so far * M^(remaining_slots) >= _TARGET? No, we need exact product.
    #     - Product of chosen so far must divide _TARGET.
    #     - The remaining product must be achievable by (k - len(chosen)) distinct factors from the allowed set (<= M, > last_chosen).
    
    # Let's use backtracking for each M.
    # To speed up, we can precompute the divisors of _TARGET.
    # And for a given M, the allowed factors are those in valid_divs that are <= M.
    
    # Let's try a direct backtracking search that minimizes the max factor naturally.
    # We pick factors in increasing order.
    # We keep track of the product so far.
    # We prune if product_so_far * (smallest possible completion) > _TARGET or product_so_far > _TARGET.
    # Actually, since we need exact product, we track remaining = _TARGET // product_so_far.
    # At each step, we pick a factor f from allowed_factors (f > last, f <= current_bound, f divides remaining).
    # If remaining == 1, we have a solution. Update best if max(factors) < best_largest.
    # If remaining > 1, we must pick more factors.
    # Pruning:
    #   - If the number of factors chosen + 1 > 12, stop.
    #   - If remaining is prime and > last, we can take it as the last factor.
    #   - If remaining is composite, we must split it.
    #   - If the smallest prime factor of remaining is <= last, we might still be able to form factors > last?
    #     No, if all prime factors of remaining are <= last, then any factor formed from them will be <= last? Not necessarily.
    #     Example: remaining = 2*17 = 34. last = 15. 34 > 15. So we can take 34.
    #     But if remaining = 2*3 = 6, last = 5. 6 > 5. We can take 6.
    #     So this pruning is not straightforward.
    
    # Let's just do a clean backtracking with the constraint that factors are picked in increasing order.
    # This avoids permutations and ensures distinctness.
    
    def backtrack(remaining, last, current, depth):
        nonlocal best_solution, best_largest
        if _remaining_time() <= 0:
            return
        
        if remaining == 1:
            if 1 <= len(current) <= _N:
                mx = max(current) if current else 0
                if mx < best_largest:
                    best_largest = mx
                    best_solution = list(current)
            return
        
        if depth > _N:
            return
            
        # If remaining is prime, we can only take it as a single factor if it's > last
        # Check if remaining is prime
        is_prime = True
        if remaining < 2:
            is_prime = False
        else:
            for i in range(2, int(remaining**0.5) + 1):
                if remaining % i == 0:
                    is_prime = False
                    break
        if is_prime:
            if remaining > last:
                current.append(remaining)
                backtrack(1, remaining, current, depth + 1)
                current.pop()
            return
        
        # Find all divisors of remaining that are > last and <= best_largest (pruning)
        # Also, they must be in valid_divs (which they are, since they divide _TARGET)
        # We iterate through valid_divs to find candidates.
        # To optimize, we can iterate through divisors of remaining directly.
        
        # Get divisors of remaining that are > last
        candidates = []
        for d in valid_divs:
            if d <= last:
                continue
            if d > remaining:
                break  # valid_divs is sorted
            if remaining % d == 0:
                # Pruning: if d >= best_largest, we can stop early? No, we might have a better solution with smaller max later?
                # Actually, if d >= best_largest, then any solution using d will have max >= d >= best_largest, so it won't improve.
                # But we might have a solution with the same max but fewer factors? The objective is to minimize max, so ties don't matter for the score.
                # So we can prune if d >= best_largest.
                if d >= best_largest:
                    break
                candidates.append(d)
        
        for d in candidates:
            current.append(d)
            backtrack(remaining // d, d, current, depth + 1)
            current.pop()
            if _remaining_time() <= 0:
                return

    # Start the search
    # We need to ensure we explore all possibilities.
    # The initial call: remaining = _TARGET, last = _N (so first factor > 12), current = [], depth = 0
    backtrack(_TARGET, _N, [], 0)

    if best_solution is not None:
        return best_solution

    # Fallback (should not happen)
    return [factorial(_N)]


if __name__ == "__main__":
    r = construct()
    print(len(r), r, max(r) if r else None)
    print("valid:", _is_valid(r))
