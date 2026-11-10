"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, prod
from itertools import combinations
from functools import lru_cache

def _prime_factorize(n):
    """Return dict of prime -> exponent for n."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 12
    target = factorial(N)
    
    # Get prime factorization of 12!
    target_factors = _prime_factorize(target)
    primes = sorted(target_factors.keys())
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Convert target to tuple of exponents
    target_vec = tuple(target_factors.get(p, 0) for p in primes)
    
    # Generate all valid factors: integers > 12 whose prime exponents don't exceed target
    # A valid factor must only use primes that divide 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # So valid factors are of form 2^a * 3^b * 5^c * 7^d * 11^e
    # where 0<=a<=10, 0<=b<=5, 0<=c<=2, 0<=d<=1, 0<=e<=1
    # and the value > 12 and <= some upper bound
    
    # Upper bound for any factor: at most 12! / 13 (smallest other factor > 12 is 13)
    # Actually, if we have k factors all > 12, the largest is at most 12! / (13^(k-1))
    # For k=1: max is 12!
    # For k=2: max is 12!/13
    # etc.
    # We'll search for solutions with different numbers of factors.
    
    # Generate all candidate factors > 12 with valid prime exponents
    candidates = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * (11 ** e)
                        if val > N and val <= target:
                            candidates.append(val)
    
    # Sort candidates by value
    candidates.sort()
    
    # For each candidate, compute its exponent vector
    cand_vecs = {}
    for val in candidates:
        f = _prime_factorize(val)
        vec = tuple(f.get(p, 0) for p in primes)
        cand_vecs[val] = vec
    
    # We want to find a subset of distinct candidates whose exponent vectors sum to target_vec
    # and minimize the maximum value in the subset.
    
    # Strategy: Binary search on the maximum factor value M.
    # For a given M, check if there exists a subset of candidates with values <= M
    # that sums to target_vec, using at most 12 factors.
    
    # Since the number of candidates is manageable, we can use DP/memoization.
    # State: (remaining_vec, num_factors_used) -> whether achievable
    # But remaining_vec is a tuple of 5 ints, which could be large.
    
    # Alternative: Use DFS with pruning.
    # Sort candidates by value. For a given max value M, only consider candidates <= M.
    
    def can_achieve(max_val, deadline):
        """Check if target_vec can be achieved using distinct factors <= max_val, at most 12 factors."""
        if time.time() > deadline:
            return None  # timeout
        
        # Filter candidates
        valid_cands = [c for c in candidates if c <= max_val]
        valid_cands.sort()
        
        # Precompute vectors
        vecs = [(c, cand_vecs[c]) for c in valid_cands]
        
        # DFS with memoization
        # State: (index, remaining_vec_tuple, count_used)
        # Use lru_cache or manual memo
        
        memo = {}
        
        def dfs(idx, rem_vec, count):
            key = (idx, rem_vec, count)
            if key in memo:
                return memo[key]
            
            if time.time() > deadline:
                memo[key] = False
                return False
            
            if all(x == 0 for x in rem_vec):
                memo[key] = True
                return True
            
            if idx >= len(vecs) or count >= 12:
                memo[key] = False
                return False
            
            # Option 1: skip vecs[idx]
            if dfs(idx + 1, rem_vec, count):
                memo[key] = True
                return True
            
            # Option 2: use vecs[idx] if it doesn't exceed rem_vec
            c_val, c_vec = vecs[idx]
            new_rem = tuple(rem_vec[i] - c_vec[i] for i in range(len(primes)))
            if all(x >= 0 for x in new_rem):
                if dfs(idx + 1, new_rem, count + 1):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        return dfs(0, target_vec, 0)
    
    # Binary search on max factor value
    lo = 13  # minimum possible max factor (if we could use 13, but 13 doesn't divide 12!)
    hi = target  # maximum possible (single factor)
    
    # Actually, the minimum possible max factor is at least 13, but 13 is prime and doesn't divide 12!,
    # so it can't be a factor. Let's find the smallest valid candidate.
    min_candidate = min(candidates) if candidates else 13
    
    best_solution = None
    best_max = target + 1
    
    # First, let's try to find any solution, then optimize.
    # Use binary search to find the minimum max factor.
    
    # But first, let's verify that a solution exists with max = target (trivial: [12!])
    # We need distinct factors > 12. [12!] has one factor, which is > 12. Valid.
    
    # Let's do a more careful search.
    # Instead of pure binary search (which might miss some values), let's iterate over possible max values
    # from the candidate set, but that's too many.
    
    # Better: binary search over the sorted candidate values.
    unique_cand_vals = sorted(set(candidates))
    
    lo_idx = 0
    hi_idx = len(unique_cand_vals) - 1
    best_idx = hi_idx  # worst case: use the largest candidate
    
    while lo_idx <= hi_idx and time.time() < deadline:
        mid_idx = (lo_idx + hi_idx) // 2
        max_val = unique_cand_vals[mid_idx]
        
        result = can_achieve(max_val, deadline)
        if result is None:  # timeout
            break
        if result:
            best_idx = mid_idx
            hi_idx = mid_idx - 1
        else:
            lo_idx = mid_idx + 1
    
    # Now extract the actual solution for best_idx
    if best_idx is not None and best_idx < len(unique_cand_vals):
        max_val = unique_cand_vals[best_idx]
        valid_cands = [c for c in candidates if c <= max_val]
        valid_cands.sort()
        vecs = [(c, cand_vecs[c]) for c in valid_cands]
        
        memo = {}
        
        def dfs_extract(idx, rem_vec, count, path):
            key = (idx, rem_vec, count)
            if key in memo:
                return memo[key]
            
            if time.time() > deadline:
                memo[key] = None
                return None
            
            if all(x == 0 for x in rem_vec):
                memo[key] = list(path)
                return list(path)
            
            if idx >= len(vecs) or count >= 12:
                memo[key] = None
                return None
            
            # Try using vecs[idx] first (to prefer smaller factors)
            c_val, c_vec = vecs[idx]
            new_rem = tuple(rem_vec[i] - c_vec[i] for i in range(len(primes)))
            if all(x >= 0 for x in new_rem):
                path.append(c_val)
                result = dfs_extract(idx + 1, new_rem, count + 1, path)
                path.pop()
                if result is not None:
                    memo[key] = result
                    return result
            
            # Try skipping
            result = dfs_extract(idx + 1, rem_vec, count, path)
            memo[key] = result
            return result
        
        solution = dfs_extract(0, target_vec, 0, [])
        if solution is not None and len(solution) >= 1 and len(solution) <= 12:
            # Verify
            if all(s > N for s in solution) and len(set(solution)) == len(solution):
                if prod(solution) == target:
                    best_solution = solution
                    best_max = max(solution)
    
    # If no solution found via binary search, fall back to trivial
    if best_solution is None:
        best_solution = [target]
        best_max = target
    
    return best_solution

# EVOLVE-BLOCK-END
