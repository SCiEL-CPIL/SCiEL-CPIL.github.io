"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import factorial, gcd, prod
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)
    MIN_VAL = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Precompute prime factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Let's verify
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    target_pf = prime_factorization(N)
    
    # We need to find distinct integers > 12 whose product is N
    # Minimize the largest factor
    
    # Strategy: Search for factorizations with varying number of factors
    # For each number of factors k (from 2 to 12), try to find k distinct factors > 12
    # whose product is N, minimizing the maximum
    
    # Key insight: If we have k factors, their geometric mean is N^(1/k)
    # To minimize the max, we want factors to be as balanced as possible
    # But they must be distinct integers > 12
    
    # Let's use a recursive search with pruning
    # Sort potential factors in ascending order to ensure distinctness
    
    # First, let's find all divisors of N that are > 12
    # But that could be a lot. Let's be smarter.
    
    # For k factors with product N, the smallest possible max is at least ceil(N^(1/k))
    # But they must be distinct and > 12
    
    # Let's try k from 2 to 12, and for each k, do a bounded search
    
    best_solution = None
    best_max = N  # single factor case
    
    # Start with the trivial solution: [N]
    # But wait, N = 479001600, which is > 12, so [N] is valid
    # But we can probably do much better
    
    # Let's try to find factorizations with more factors
    
    # For a given k, we want k distinct integers a1 < a2 < ... < ak
    # all > 12, with product N, minimizing ak
    
    # Lower bound on ak: if a1 >= 13, a2 >= 14, ..., ak-1 >= 13+(k-2), 
    # then ak >= N / prod(13, 14, ..., 13+k-2)
    # But this is a rough bound
    
    # Let's do a DFS with pruning
    def dfs(remaining, min_factor, current_factors, depth, max_depth):
        nonlocal best_solution, best_max
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if depth == max_depth:
            if remaining == 1:
                # Check if all factors are distinct and > 12
                if len(set(current_factors)) == len(current_factors):
                    if all(f > 12 for f in current_factors):
                        max_f = max(current_factors)
                        if max_f < best_max:
                            best_max = max_f
                            best_solution = current_factors[:]
            return
        
        if remaining == 1:
            return
        
        # Pruning: if we can't fit the remaining factors
        # The remaining product must be achievable with (max_depth - depth) factors
        # each >= min_factor, distinct from current_factors
        
        # Lower bound: the smallest possible product for remaining factors
        # is min_factor * (min_factor+1) * ... * (min_factor + (max_depth-depth-1))
        # But we need to skip factors already used
        # This is complex, so let's use a simpler bound
        
        # Upper bound on the next factor: 
        # If we pick factor f, then remaining/f must be factorizable by 
        # (max_depth - depth - 1) factors each > f
        # So f * (f+1) * ... * (f + max_depth-depth-2) <= remaining
        # This gives an upper bound on f
        
        # Also, f must divide remaining
        # And f > min_factor
        # And f not in current_factors
        
        # Let's iterate through possible divisors of remaining that are > min_factor
        
        # To keep it efficient, let's limit the range
        # The next factor f must satisfy:
        # f > min_factor
        # f divides remaining
        # remaining / f must be factorable into (max_depth - depth - 1) distinct factors > f
        
        # Simple bound: f * (f+1) * ... * (f + max_depth-depth-2) <= remaining
        # This is a rough bound assuming consecutive integers
        
        # Let's compute the maximum possible f
        # For the last factor (when depth == max_depth - 1), f = remaining
        # For earlier, f must be such that remaining/f >= (f+1)(f+2)...(f+max_depth-depth-2)
        
        # Let's just iterate through divisors up to a reasonable limit
        
        # Find divisors of remaining that are > min_factor
        # But remaining can be large, so let's be careful
        
        # Limit: f must be at most remaining / ((min_factor+1) * ... ) but this is complex
        # Let's use: f <= remaining ** (1 / (max_depth - depth))
        # Because if we have m remaining factors, each at least f, then f^m <= remaining
        
        m_remaining = max_depth - depth
        if m_remaining == 1:
            # Last factor
            f = remaining
            if f > min_factor and f > 12 and f not in current_factors:
                new_factors = current_factors + [f]
                if len(set(new_factors)) == len(new_factors) and all(x > 12 for x in new_factors):
                    max_f = max(new_factors)
                    if max_f < best_max:
                        best_max = max_f
                        best_solution = new_factors[:]
            return
        
        # For m_remaining > 1
        # f must satisfy f^m_remaining <= remaining, so f <= remaining ** (1/m_remaining)
        max_f_possible = int(remaining ** (1.0 / m_remaining)) + 2
        
        # Also f > min_factor
        # And f must divide remaining
        
        # Let's iterate f from min_factor+1 to min(max_f_possible, some_limit)
        # But we need f to divide remaining
        
        # To be efficient, let's find divisors of remaining in range [min_factor+1, max_f_possible]
        # But remaining might have many divisors
        
        # Alternative: iterate f and check if remaining % f == 0
        # Limit the range to keep it fast
        
        # Additional pruning: the remaining factors after f must all be > f
        # So their product >= (f+1)(f+2)...(f+m_remaining-1)
        # So remaining/f >= (f+1)(f+2)...(f+m_remaining-1)
        # This gives a tighter upper bound on f
        
        # Let's compute this bound
        # prod = (f+1)(f+2)...(f+m_remaining-1)
        # remaining/f >= prod
        # remaining >= f * prod
        
        # This is implicit in f, so let's just iterate
        
        # To limit the search space, cap max_f_possible
        # If max_f_possible is too large, we might have trouble
        # But for our problem, N = 12! ≈ 4.8e8, so factors won't be too large
        
        # Let's iterate f from min_factor+1 to min(max_f_possible, 10000)
        # But 10000 might be too small... let's think
        # 12! = 479001600
        # If we have 2 factors, each around sqrt(479001600) ≈ 21886
        # So we need to go up to at least ~22000 for 2 factors
        # For 3 factors, each around (479001600)^(1/3) ≈ 782
        # So the bound depends on m_remaining
        
        # Let's use max_f_possible but cap it at reasonable values
        # And add a time check
        
        upper = min(max_f_possible, 50000)
        
        for f in range(min_factor + 1, upper + 1):
            if time.time() > deadline:
                return
            
            if remaining % f != 0:
                continue
            
            if f in current_factors:
                continue
            
            # Additional pruning: remaining/f must be factorable into m_remaining-1 factors > f
            # The minimum product of m_remaining-1 distinct integers > f is
            # (f+1)(f+2)...(f+m_remaining-1)
            # So remaining/f >= (f+1)(f+2)...(f+m_remaining-1)
            
            # Compute this minimum product
            min_prod = 1
            for i in range(1, m_remaining):
                min_prod *= (f + i)
                if min_prod > remaining // f:
                    break
            
            if min_prod > remaining // f:
                # Even the smallest possible remaining factors are too large
                # And since f is increasing, this will only get worse
                # Actually no, as f increases, min_prod increases, but remaining/f decreases
                # So if this fails, larger f will also fail
                break
            
            new_factors = current_factors + [f]
            dfs(remaining // f, f, new_factors, depth + 1, max_depth)
    
    # Try different numbers of factors
    # Start with more factors (likely to give smaller max)
    for k in range(MAX_FACTORS, 1, -1):
        if time.time() > deadline:
            break
        
        # Lower bound check: can we even have k factors?
        # The minimum product of k distinct integers > 12 is 13*14*...*13+k-1
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
        
        if min_possible > N:
            continue
        
        # Upper bound on max factor for k factors:
        # If all factors are at least 13, 14, ..., 13+k-1, the max is at least 13+k-1
        # But we want to minimize the max
        
        # Start the DFS
        dfs(N, 12, [], 0, k)
        
        # Early termination if we found a very good solution
        # The theoretical minimum max for k factors is at least ceil(N^(1/k))
        # If best_max is close to this, we can stop
        lower_bound = int(N ** (1.0 / k)) + 1
        if best_solution and best_max <= lower_bound + 5:
            break
    
    if best_solution is None:
        return [N]
    
    return best_solution

# EVOLVE-BLOCK-END
