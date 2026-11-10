"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache
import bisect

TARGET = factorial(12)  # 479001600
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11

def _all_divisors():
    """Generate all divisors of TARGET, sorted."""
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    return divisors

def _build_valid_divisors(max_val):
    """Generate all divisors of TARGET that are > 12 and <= max_val, sorted."""
    all_d = _all_divisors()
    result = [d for d in all_d if 12 < d <= max_val]
    return result

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 90s kill

    # Get all valid divisors (>12), sorted ascending
    all_divs = _build_valid_divisors(TARGET)
    n_divs = len(all_divs)
    
    best_result = [TARGET]  # fallback: single factor
    best_max = TARGET

    # Strategy: We want to minimize the largest factor.
    # Key insight: to minimize the max factor, we want to use as many factors as possible
    # (up to 12), each as small as possible.
    # 
    # Approach: Try to build a solution greedily.
    # For a given set of allowed divisors (those <= some bound B), find a subset of distinct
    # divisors (each > 12, <= B) whose product is TARGET, using 1-12 factors.
    #
    # Better approach: Directly search for the optimal decomposition.
    # We'll use DFS that tries to pick factors in increasing order, aiming to keep
    # each factor small.
    #
    # Even better: Since we want to minimize the max, let's think of it as:
    # What is the smallest B such that there exists a valid decomposition with all factors <= B?
    #
    # We can binary search on B, but the search space of B is the divisors of TARGET.
    # For each candidate B, we check feasibility.
    #
    # For feasibility check: Given a set S of allowed divisors (all in (12, B]),
    # does there exist a subset of S (distinct, size 1-12) whose product is TARGET?
    #
    # This is a subset product problem. With |S| potentially large, we need pruning.
    #
    # Alternative: Direct constructive search.
    # Since TARGET has a known prime factorization, we can do a DFS over the "remaining"
    # prime exponents, trying to assign them to factors.
    
    # Let's try a direct approach: DFS where we build factors one by one.
    # At each step, we have a "remaining" value (divisor of TARGET) and a set of used divisors.
    # We try to pick the smallest unused divisor d > 12, d <= remaining, d | remaining.
    # We want to minimize the max factor, so we should try smaller d first.
    # But we also need to ensure the remaining can be completed.
    
    # Actually, let's think differently. The optimal solution likely uses many factors.
    # Let's try to find a solution with exactly k factors for k = 12, 11, ..., 1.
    # For each k, find the decomposition that minimizes the max factor.
    # The answer is the min over all k of (min max factor for k-factor decomposition).
    
    # For a fixed k, to minimize the max factor, we want the factors to be as balanced
    # as possible (all close to TARGET^(1/k)).
    
    # Let's use a recursive search with strong pruning.
    # State: (idx in divs list, remaining product, factors_used, current_max)
    # We want to find any completion, and among all, minimize the max.
    
    # Better: Let's do a branch-and-bound.
    # We'll search for a decomposition. The DFS picks factors in increasing order
    # (to ensure distinctness and avoid permutations).
    # We prune if current_max is already >= best_max.
    # We also prune if remaining product can't be achieved with the remaining number of factors.
    
    # Let me implement a clean DFS:
    
    def dfs(start_idx, remaining, used_count, current_max, best):
        """
        Try to decompose 'remaining' into (12 - used_count) more distinct factors,
        each > 12, each >= all_divs[start_idx], each <= best_max (to improve).
        Returns the list of additional factors or None.
        """
        nonlocal best_max, best_result
        elapsed = time.time() - start
        if elapsed > deadline - 2.0:
            return None
        
        if remaining == 1:
            if used_count >= 1 and used_count <= 12:
                return []
            return None
        
        if used_count >= 12:
            return None
        
        # Pruning: if remaining is a valid single factor (distinct, > 12, not used)
        # and remaining <= best_max, we can take it as the last factor.
        # But we need to check it's not already used. We track used via the divs index.
        
        # Lower bound on number of more factors needed: 1
        # Upper bound: 12 - used_count
        
        # If remaining itself is > 12, not used, and <= best_max, it could be the last factor.
        # But we need to ensure it's not in the used set. Since we pick in increasing order
        # from all_divs, and all_divs is sorted, if we've picked indices < start_idx,
        # then remaining (if in all_divs) must be at index >= start_idx to be "new".
        
        # Let's just iterate.
        for i in range(start_idx, n_divs):
            d = all_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            # d is a valid next factor
            new_max = max(current_max, d)
            if new_max >= best_max:
                # Can't improve, and since divs are sorted, further d will be >= d,
                # so new_max won't decrease. But wait, we might still find a solution
                # with the same max but we want strictly better. If new_max >= best_max,
                # skip.
                # Actually, if new_max == best_max, we don't improve. Skip.
                continue
            
            sub = dfs(i + 1, remaining // d, used_count + 1, new_max, best)
            if sub is not None:
                return [d] + sub
            # If we used up time, bail
            if time.time() - start > deadline - 2.0:
                return None
        
        return None

    # Start the search
    # Initial: remaining = TARGET, used_count = 0, current_max = 0 (or -inf)
    # We want to find any valid decomposition, then improve.
    
    # First, get a quick valid solution: [TARGET] itself (1 factor)
    # Then try to improve.
    
    # Let's start with a greedy heuristic to get a good initial upper bound.
    # Greedy: repeatedly pick the smallest valid divisor of the remaining product.
    
    def greedy_decompose():
        remaining = TARGET
        factors = []
        used = set()
        for d in all_divs:
            if d > remaining:
                break
            if remaining % d == 0 and d not in used:
                # Check if we can complete: remaining // d must be decomposable
                # For now, just take it if it helps
                factors.append(d)
                used.add(d)
                remaining //= d
                if remaining == 1:
                    break
                if len(factors) >= 12:
                    break
        if remaining == 1 and 1 <= len(factors) <= 12:
            return factors
        return None
    
    greedy_sol = greedy_decompose()
    if greedy_sol:
        best_max = max(greedy_sol)
        best_result = greedy_sol
    
    # Now do DFS to find better solutions
    # The DFS above tries to find a solution with max < best_max.
    # Let's call it.
    
    # We need to track best_max and best_result as nonlocal or use a wrapper.
    # Let me restructure.
    
    search_state = {'best_max': best_max, 'best_result': best_result}
    
    def dfs2(start_idx, remaining, used_count, current_max):
        nonlocal search_state
        elapsed = time.time() - start
        if elapsed > deadline - 2.0:
            return None
        
        if remaining == 1:
            if 1 <= used_count <= 12:
                return []
            return None
        
        if used_count >= 12:
            return None
        
        for i in range(start_idx, n_divs):
            d = all_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= search_state['best_max']:
                continue
            
            sub = dfs2(i + 1, remaining // d, used_count + 1, new_max)
            if sub is not None:
                sol = [d] + sub
                search_state['best_max'] = new_max
                search_state['best_result'] = sol
                return sol
            
            if time.time() - start > deadline - 2.0:
                return None
        
        return None
    
    # Call DFS
    dfs2(0, TARGET, 0, 0)
    
    return search_state['best_result']

# EVOLVE-BLOCK-END
