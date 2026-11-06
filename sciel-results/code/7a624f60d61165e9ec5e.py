import math
import time

def construct():
    start = time.time()
    deadline = start + 80  # 80s safety margin
    
    target_val = math.factorial(12)  # 479001600
    primes = [2, 3, 5, 7, 11]
    
    def get_vec(n):
        v = [0]*5
        for idx, p in enumerate(primes):
            while n % p == 0:
                v[idx] += 1
                n //= p
        return tuple(v)
    
    target_vec = get_vec(target_val)
    
    # Generate all divisors of target_val
    divisors = set()
    i = 1
    while i * i <= target_val:
        if target_val % i == 0:
            divisors.add(i)
            divisors.add(target_val // i)
        i += 1
    
    # Filter: > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    div_vecs = {d: get_vec(d) for d in valid_divs}
    
    # Precompute: for quick "can the remaining vector be formed" we could do memoization,
    # but let's first try direct DFS with pruning.
    
    # We'll binary search on M (the max factor index in valid_divs).
    # check(M_idx): is there a subset of valid_divs[0..M_idx] (distinct, all >12) 
    #   with product vector == target_vec, size 1..12?
    
    n_divs = len(valid_divs)
    
    # Memoize: (pos, vec_remaining, count_used) -> bool
    # But vec_remaining is a tuple, count_used small. Use lru_cache.
    from functools import lru_cache
    
    # To make it work with binary search, we need the candidate set to depend on M.
    # Instead, let's do: for a fixed M_idx, candidates = valid_divs[0..M_idx].
    # DFS(pos, cur_vec, count): 
    #   remaining = target_vec - cur_vec
    #   if remaining == (0,0,0,0,0) and count >= 1: return True
    #   if count == 12: return False
    #   if pos >= len(candidates): return False
    #   # Prune: the remaining vector must be formable by factors in candidates[pos..]
    #   # Lower bound: even the smallest available factor... 
    #   for i in range(pos, len(candidates)):
    #       d = candidates[i]
    #       v = div_vecs[d]
    #       new_vec = tuple(cur_vec[j] + v[j] for j in range(5))
    #       if any(new_vec[j] > target_vec[j] for j in range(5)):
    #           continue
    #       if dfs(i+1, new_vec, count+1):
    #           return True
    #   return False
    
    # This might be slow without memoization. Let's add memoization on (pos, cur_vec, count).
    # But cur_vec has many possible values. Let's use a dict.
    
    # Actually, let me think about the search space more carefully.
    # The key insight: we want to MINIMIZE the max. So we should search for the smallest M
    # that works. Binary search on M.
    
    # Let me implement check with memoization, and binary search.
    
    def check(M_idx):
        candidates = valid_divs[:M_idx+1]
        cvecs = [div_vecs[d] for d in candidates]
        n = len(candidates)
        
        memo = {}
        
        def dfs(pos, cur_vec, count):
            # cur_vec is tuple
            remaining = tuple(target_vec[j] - cur_vec[j] for j in range(5))
            if remaining == (0,0,0,0,0):
                return count >= 1
            if count == 12:
                return False
            if pos >= n:
                return False
            
            key = (pos, cur_vec, count)
            if key in memo:
                return memo[key]
            
            # Prune: remaining must be non-negative (guaranteed by construction)
            # Prune: the minimum product we can add is candidates[pos] (smallest available)
            #   so remaining vector must be >= vec(candidates[pos]) in some sense? Not directly.
            # Instead: remaining as integer = product of remaining primes^exponents.
            #   The smallest factor we can pick is candidates[pos]. So remaining_int >= candidates[pos].
            #   remaining_int = prod(target_primes^remaining_exp)
            rem_int = 1
            for j, p in enumerate(primes):
                rem_int *= p ** remaining[j]
            
            # If rem_int < candidates[pos], we can't even pick one more factor (smallest available)
            # Actually we need rem_int to be formable by factors each >= candidates[pos] and > last picked.
            # But candidates[pos] is the smallest in the remaining pool. If rem_int < candidates[pos], 
            #   we can't form it (since each factor >= candidates[pos] > rem_int, product would be > rem_int 
            #   unless we use 0 factors, but rem_int > 1 here since remaining != 0).
            # Wait, if remaining != 0 then rem_int > 1. And each factor >= candidates[pos]. 
            #   If candidates[pos] > rem_int, then even one factor is too big. Fail.
            if rem_int < candidates[pos]:
                memo[key] = False
                return False
            
            # Also, rem_int must be <= product of the largest (12-count) factors in candidates[pos..]
            #   (since we can use at most 12-count more factors, each <= candidates[-1])
            #   max possible = candidates[-1] ** (12-count)  (loose)
            #   Better: product of the (12-count) largest in candidates[pos..]
            max_avail = 12 - count
            if max_avail <= 0:
                memo[key] = False
                return False
            # Product of largest max_avail in candidates[pos..]
            largest = candidates[max(0, n - max_avail):]  # not exactly, should be from pos..
            # Actually the largest factors overall are at the end. The largest max_avail available 
            #   from pos.. are candidates[n-max_avail:] if n-max_avail >= pos, else candidates[pos:].
            start_idx = max(pos, n - max_avail)
            if start_idx < pos:
                start_idx = pos
            prod_max = 1
            for x in candidates[start_idx:]:
                prod_max *= x
            # But we might use fewer than max_avail factors. The max product with up to max_avail factors 
            #   is the product of the largest max_avail. If rem_int > prod_max, fail.
            if rem_int > prod_max:
                memo[key] = False
                return False
            
            for i in range(pos, n):
                v = cvecs[i]
                new_vec = tuple(cur_vec[j] + v[j] for j in range(5))
                if any(new_vec[j] > target_vec[j] for j in range(5)):
                    continue
                if dfs(i+1, new_vec, count+1):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        return dfs(0, (0,0,0,0,0), 0)
    
    # Binary search on M_idx
    lo = 0
    hi = n_divs - 1
    best_M_idx = None
    
    # First, verify a solution exists (should, trivially with [target_val] if target_val > 12)
    # target_val = 479001600 > 12, so [target_val] is a valid 1-factor solution.
    # But we want to minimize max, so we search for the smallest M.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if check(mid):
            best_M_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_M_idx is None:
        # Fallback: return [target_val]
        return [target_val]
    
    # Now reconstruct the actual factors for M = valid_divs[best_M_idx]
    M = valid_divs[best_M_idx]
    candidates = valid_divs[:best_M_idx+1]
    cvecs = [div_vecs[d] for d in candidates]
    n = len(candidates)
    
    memo = {}
    def dfs_recon(pos, cur_vec, count, path):
        remaining = tuple(target_vec[j] - cur_vec[j] for j in range(5))
        if remaining == (0,0,0,0,0):
            if count >= 1:
                return True
            return False
        if count == 12:
            return False
        if pos >= n:
            return False
        
        key = (pos, cur_vec, count)
        if key in memo:
            return memo[key]
        
        rem_int = 1
        for j, p in enumerate(primes):
            rem_int *= p ** remaining[j]
        
        if rem_int < candidates[pos]:
            memo[key] = False
            return False
        
        max_avail = 12 - count
        if max_avail <= 0:
            memo[key] = False
            return False
        start_idx = max(pos, n - max_avail)
        if start_idx < pos:
            start_idx = pos
        prod_max = 1
        for x in candidates[start_idx:]:
            prod_max *= x
        if rem_int > prod_max:
            memo[key] = False
            return False
        
        for i in range(pos, n):
            v = cvecs[i]
            new_vec = tuple(cur_vec[j] + v[j] for j in range(5))
            if any(new_vec[j] > target_vec[j] for j in range(5)):
                continue
            path.append(candidates[i])
            if dfs_recon(i+1, new_vec, count+1, path):
                memo[key] = True
                return True
            path.pop()
        
        memo[key] = False
        return False
    
    path = []
    if not dfs_recon(0, (0,0,0,0,0), 0, path):
        return [target_val]
    
    # Verify
    prod = 1
    for d in path:
        prod *= d
    assert prod == target_val, f"Product mismatch: {prod} != {target_val}"
    assert len(set(path)) == len(path), "Not distinct"
    assert all(d > 12 for d in path), "Factor not > 12"
    assert 1 <= len(path) <= 12, "Bad count"
    
    return path
