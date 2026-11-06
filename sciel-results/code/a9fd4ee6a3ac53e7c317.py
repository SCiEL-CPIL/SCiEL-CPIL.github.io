import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s
    
    N = math.factorial(12)
    
    # Precompute prime factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^2 * 11
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 2, 1]
    
    # Generate all divisors of N
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    divisors.sort()
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    # We want to find a set of distinct divisors d1, ..., dk (k<=12)
    # such that product is N, and max(di) is minimized.
    # All di > 12.
    
    # Strategy: 
    # Since we want to minimize the max factor, we can try to find a solution
    # where the largest factor is as small as possible.
    # Let M be the largest factor. Then all factors are in (12, M].
    # Also, the product is N.
    
    # We can iterate over possible values of M?
    # M must be a divisor of N.
    # And M > 12.
    # We can sort divisors and try to find the smallest M such that
    # N can be decomposed into distinct factors in (12, M] with product N.
    
    # This is a subset product problem (with distinctness and constraints).
    # But note: the factors don't have to be a subset of divisors?
    # Yes, they must be divisors of N because their product is N.
    
    # So, for a given M, we have a set S_M = {d in valid_divs | d <= M}.
    # We want to know if there is a subset of S_M with product N.
    # Since all elements are distinct and > 1, and product is fixed,
    # this is a unique factorization problem? No, but the set of factors is determined by the partition of prime exponents.
    
    # Actually, a simpler approach:
    # Use DFS to find a decomposition.
    # To minimize the max, we can use a branch and bound.
    # Or, we can try to construct the solution greedily?
    
    # Let's try a DFS that picks factors in increasing order.
    # To minimize the max, we should try to keep factors small.
    # But we also need to ensure the remaining product can be formed by factors > current.
    
    # Better: iterate M from smallest possible divisor > 12 upwards.
    # For each M, check if N can be formed by distinct divisors in (12, M].
    # The first M that works gives the optimal solution.
    
    # How to check for a given M?
    # Backtracking search for a subset of S_M with product N.
    # Since N is large, but the number of factors is small (<=12),
    # and the divisors are constrained, this might be feasible.
    
    # However, checking every M might be slow if the optimal M is large.
    # But the optimal M is likely not too large.
    # 12! = 479001600.
    # If we use 12 factors, the geometric mean is (479001600)^(1/12) ~ 10.
    # So factors around 10-20 are likely.
    # Since factors must be > 12, they are likely in the range 13-30.
    
    # Let's find the smallest divisor of N that is > 12.
    # min_div = 13? No, 13 is not a divisor of 12!.
    # Divisors of 12!: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, ...
    # 13 is not a divisor.
    # 14 = 2*7.
    # 15 = 3*5.
    # 16 = 2^4.
    # 18 = 2*3^2.
    # 20 = 2^2*5.
    # 21 = 3*7.
    # 22 = 2*11.
    # 24 = 2^3*3.
    # 25 = 5^2.
    # 28 = 2^2*7.
    # 30 = 2*3*5.
    # 33 = 3*11.
    # 35 = 5*7.
    # 36 = 2^2*3^2.
    # ...
    
    # So we can iterate M over valid_divs.
    
    # To speed up the subset product check:
    # We can use a dictionary or set to track reachable products?
    # No, product N is huge.
    
    # Backtracking is the way.
    # Sort S_M.
    # Pick the largest possible factors first? Or smallest?
    # To find *a* solution, any order works.
    # But to prune, if the product of the remaining available factors (largest ones) is less than N/current_product, we can prune.
    # Also, if current_product > N, prune.
    
    # Let's implement a helper function `can_decompose(N, allowed_set, current_product, start_idx)`.
    # But we need to return the actual factors too.
    
    # Alternative:
    # Just do the main DFS once, but with a strong pruning:
    # We want to minimize the max factor.
    # So we can sort the divisors and try to pick them in a way that keeps the max low.
    # Actually, the "iterate M" approach is cleaner for optimality.
    
    # Let's try the "iterate M" approach.
    
    # Optimization:
    # If we pick a factor d, the next factors must be > d.
    # So in the subset check for a given M, we should pick factors in increasing order.
    
    # Let's write the function to find a decomposition for a given M.
    
    def find_decomp_for_M(M):
        # Returns a list of factors if found, else None.
        # Factors must be distinct, > 12, <= M, product N.
        allowed = [d for d in valid_divs if d <= M]
        # Sort allowed
        allowed.sort()
        
        # Backtracking
        # state: remaining product, next index in allowed
        # We want to pick a subset of allowed with product N.
        # To ensure distinctness, we just pick each element at most once.
        
        # Pruning:
        # 1. If remaining == 1, success (if we picked at least 1? No, product 1 means no factors, but we need product N. So if N==1, but N!=1).
        #    Actually, if remaining == 1, we have found a valid set of factors.
        # 2. If no more elements, fail.
        # 3. If product of all remaining allowed elements < remaining, fail.
        #    (Since all factors are >= 1, max possible product is product of all remaining).
        
        # This pruning (3) is strong.
        
        # Let's precompute suffix products of allowed?
        # suffix_prod[i] = product of allowed[i:]
        # If suffix_prod[i] < remaining, then we cannot reach 'remaining' using allowed[i:].
        
        # But product can be huge. Python handles big ints.
        
        def backtrack(remaining, idx):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= len(allowed):
                return None
            
            # Prune: if product of allowed[idx:] < remaining
            # We can compute this on the fly or precompute.
            # Precomputing for each M is expensive if M changes.
            # But allowed changes with M.
            
            # Let's just iterate.
            # To speed up, we can pick the largest factors first?
            # If we pick large factors, we reduce remaining quickly.
            # But we need distinct factors.
            
            # Let's try picking from the end of allowed (largest first) to reduce remaining faster.
            # Or from the beginning?
            # If we pick small factors, remaining stays large, and we might run out of factors.
            # Picking large factors is usually better for product reduction.
            
            # However, we must ensure the remaining product can be formed by factors > current.
            # If we pick in decreasing order, the "next" factors must be smaller.
            # This is symmetric.
            
            # Let's stick to increasing order for simplicity, but prune with suffix product.
            
            # Compute suffix product from idx to end.
            # This is O(n) per call, which is bad.
            # Precompute suffix products for the current 'allowed' list.
            
            # Since 'allowed' is fixed for a given M, we can precompute suffix products once.
            
            pass
        
        # Precompute suffix products
        suffix_prods = [1] * (len(allowed) + 1)
        for i in range(len(allowed) - 1, -1, -1):
            suffix_prods[i] = suffix_prods[i+1] * allowed[i]
        
        def backtrack(remaining, idx):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= len(allowed):
                return None
            
            # Prune: if suffix_prods[idx] < remaining, we can't reach remaining
            if suffix_prods[idx] < remaining:
                return None
            
            # Option 1: Skip allowed[idx]
            res = backtrack(remaining, idx + 1)
            if res is not None:
                return res
            
            # Option 2: Take allowed[idx]
            if remaining % allowed[idx] == 0:
                res = backtrack(remaining // allowed[idx], idx + 1)
                if res is not None:
                    return [allowed[idx]] + res
            
            return None
        
        res = backtrack(N, 0)
        if res:
            return res
        return None
    
    # Iterate M over valid_divs
    # To find the MINIMUM max factor, we iterate M in increasing order.
    # The first M for which find_decomp_for_M(M) returns a non-None result is optimal.
    
    # But wait, is it possible that a solution with max factor M exists, but find_decomp_for_M(M) fails due to search order?
    # No, it's a complete search (with pruning that is safe).
    
    # However, iterating M from smallest to largest might be slow if the optimal M is large.
    # But as estimated, optimal M is likely small (around 30-50).
    
    # Let's try.
    
    # One optimization: if M < 13, skip. (Already handled by valid_divs)
    
    # Also, we can stop if M is too large? No, we want the minimum.
    
    # Let's add a timeout check in the outer loop.
    
    # To speed up, we can skip M values that are not divisors?
    # valid_divs contains only divisors.
    
    # Let's run it.
    
    for M in valid_divs:
        if time.time() > deadline:
            break
        
        # Quick check: if M < 13, continue (already filtered)
        
        # Another prune: if M is small, the max product of k distinct factors <= M is limited.
        # But suffix_prod handles this in backtrack.
        
        res = find_decomp_for_M(M)
        if res is not None:
            # Found a solution with max factor M.
            # Since we iterate M in increasing order, this is the minimum possible max factor.
            return res
    
    # If no solution found within time, return a fallback?
    # The task guarantees a solution exists.
    # If we run out of time, we should return the best found so far?
    # But the loop breaks and returns nothing?
    # We need to return *something* valid.
    # If we didn't find one, we can return [N] (max factor N).
    # But [N] is valid (1 factor, >12, product N).
    # So we can initialize best_solution = [N] and update it.
    
    # Wait, the loop returns immediately when found.
    # If it finishes without finding, it falls through.
    # So we should return [N] as fallback.
    
    return [N]

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 1
    for x in res:
        prod *= x
    print("Product:", prod)
    print("12!:", math.factorial(12))
    print("Max:", max(res))
    print("Distinct:", len(res) == len(set(res)))
    print("All > 12:", all(x > 12 for x in res))
