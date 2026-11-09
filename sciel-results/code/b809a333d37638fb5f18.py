def construct():
    import time
    from math import factorial
    from functools import lru_cache
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Filter: divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    # We want to find a subset of distinct elements from valid_divs
    # whose product is exactly N, minimizing the max element.
    # Strategy: try max values in increasing order.
    
    # For a given max M, we can only use divisors in (12, M].
    # Use backtracking with memoization on (remaining_product, min_index)
    # where min_index ensures we pick distinct divisors in sorted order.
    
    # To make this efficient, sort valid_divs and for a given M,
    # consider only divisors <= M.
    
    # Let's precompute: for each possible max candidate,
    # which divisors are available.
    
    # Better approach: sort valid_divs. For max = valid_divs[k],
    # available set is valid_divs[0..k] (all <= valid_divs[k] and > 12).
    
    # Backtracking: pick divisors in increasing index order to avoid permutations.
    # State: (remaining, start_index) -> can we complete?
    # But remaining can be large, so use a set of (remaining, start) visited.
    
    # Actually, let's use a different approach:
    # Since the number of factors is at most 12, and we need product = N,
    # we can do a DFS that picks one divisor at a time.
    
    # Key pruning: if remaining product is 1, we're done.
    # If remaining product < 13 (smallest valid divisor), fail.
    # If remaining product is prime and > 12, it must be a single factor (but must be in our set).
    # If remaining product is composite, we can split it.
    
    # Let me use a cleaner backtracking:
    # dfs(remaining, max_allowed, used_set) -> list of factors or None
    # But used_set is expensive. Instead, enforce increasing order.
    
    # Better: dfs(remaining, min_factor, max_allowed) -> list or None
    # where all factors must be in [min_factor, max_allowed] and distinct (enforced by increasing).
    
    # Actually, let's think differently. Since we want to minimize the max,
    # let's binary search or linear scan on the max value.
    
    # For a fixed max M, the question is: can we partition N into distinct factors,
    # each in (12, M]?
    
    # Let's use a recursive search with memoization.
    # State: (remaining, min_idx) where min_idx is the index in valid_divs (filtered to <= M)
    # from which we can pick the next factor.
    
    # This might still be expensive. Let me think about the structure.
    
    # 12! = 2^10 * 3^5 * 5^2 * 7
    # The smallest factor > 12 is 13. But 13 doesn't divide 12!, so it can't be used.
    # Valid divisors > 12 of 12!: 14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 28, 30, ...
    
    # Let me just do a well-pruned DFS with a time limit.
    
    # Approach: 
    # 1. Get sorted list of valid divisors (d > 12, d | N)
    # 2. For max_candidate in sorted(valid_divs):
    #    Check if we can form N using distinct divisors from valid_divs[:k+1] where valid_divs[k] = max_candidate
    #    Use DFS with strong pruning
    # 3. Return the first (smallest max) that works
    
    # DFS(remaining, start_idx, current_factors):
    #   if remaining == 1: return current_factors
    #   if remaining < 13: return None (can't form with factors > 12)
    #   if len(current_factors) >= 12: return None
    #   for i in range(start_idx, len(available)):
    #       d = available[i]
    #       if d > max_candidate: break
    #       if remaining % d != 0: continue
    #       # Pruning: if d > remaining, break (but d <= max_candidate and we check divisibility)
    #       # Additional pruning: remaining/d must be >= 13^(remaining_factors_needed)
    #       # Simple pruning: remaining/d >= 13
    #       if remaining // d < 13 and remaining // d != 1: continue
    #       result = DFS(remaining // d, i+1, current_factors + [d])
    #       if result is not None: return result
    #   return None
    
    # This could be slow. Let me add more pruning.
    
    # Key insight: the number of factors is small (at most ~10 or so for 12!).
    # Let me estimate: if all factors are ~13, we'd need log_13(479001600) ≈ 6.5 factors.
    # So at most 12 factors is fine.
    
    # Let me implement with memoization on (remaining, start_idx) for a fixed max_candidate.
    # But the search space might be large. Let me try a different tactic.
    
    # Alternative: since we want to minimize the max, and the divisors are sorted,
    # let's try max values starting from the smallest possible.
    
    # What's the smallest possible max? We need at least 2 factors (since 12! > 12).
    # With 2 factors: a*b = 12!, a,b > 12, a < b. Minimize b.
    # b >= ceil(12!/13) ... but 13 doesn't divide 12!.
    # The smallest factor > 12 that divides 12! is 14.
    # 12!/14 = 34214400. Is 34214400 > 12? Yes. So [14, 34214400] is valid with max = 34214400.
    # But we can do better with more factors.
    
    # Let me just run the DFS with good pruning and a time limit.
    
    valid_divs_sorted = sorted(set(d for d in divisors if d > 12))
    n_divs = len(valid_divs_sorted)
    
    # For efficiency, precompute which divisors divide which remainders? Too complex.
    # Just do DFS with memoization per max_candidate.
    
    from sys import setrecursionlimit
    setrecursionlimit(10000)
    
    best_result = None
    best_max = N  # fallback: [N] itself (but N > 12, so valid)
    
    # Try increasing max candidates
    for max_idx in range(n_divs):
        if time.time() > deadline:
            break
        max_candidate = valid_divs_sorted[max_idx]
        
        # Available divisors: valid_divs_sorted[0..max_idx]
        available = valid_divs_sorted[:max_idx+1]
        
        # DFS with memoization
        # State: (remaining, start_idx)
        # Return: list of factors or None
        memo = {}
        
        def dfs(remaining, start_idx):
            if time.time() > deadline:
                return None
            if remaining == 1:
                return []
            if remaining < 13:
                return None
            key = (remaining, start_idx)
            if key in memo:
                return memo[key]
            
            # Pruning: if remaining is prime and > 12, it must be a single factor
            # Check if remaining is in available and >= available[start_idx]
            # Actually, let's just try all divisors from start_idx
            
            for i in range(start_idx, len(available)):
                d = available[i]
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                rem = remaining // d
                # Pruning: rem must be 1 or >= 13
                if rem < 13 and rem != 1:
                    continue
                # Pruning: rem must be achievable with factors > 12
                # If rem is prime, it must be >= 13 and in our available set (but we'll handle in recursion)
                result = dfs(rem, i+1)
                if result is not None:
                    memo[key] = [d] + result
                    return [d] + result
            
            memo[key] = None
            return None
        
        result = dfs(N, 0)
        if result is not None:
            best_result = result
            best_max = max_candidate
            break  # Found the minimum max
    
    if best_result is None:
        # Fallback
        best_result = [N]
    
    return best_result
