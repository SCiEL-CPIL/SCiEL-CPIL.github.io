import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 40  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target that are > 12
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    candidates = sorted([d for d in divisors if d > 12])
    val_to_idx = {c: i for i, c in enumerate(candidates)}
    n_cand = len(candidates)
    
    # Precompute which candidate indices divide each divisor
    div_to_indices = {}
    for d in divisors:
        if d <= 12:
            continue
        indices = []
        for i, c in enumerate(candidates):
            if c > d:
                break
            if d % c == 0:
                indices.append(i)
        div_to_indices[d] = indices
    
    # Memoization: key is (rem, start_idx), value is the minimum max_val that works, or None
    # Actually, for a given (rem, start_idx), if it works with max_val, it works with any larger max_val
    # So we can store the minimum max_val that works
    min_max_val = {}
    
    def can_decompose(rem, start_idx, max_val):
        if time.time() > deadline:
            return False
        
        if rem == 1:
            return True
        
        if rem < 13:
            return False
        
        # If rem itself is a valid factor
        if 12 < rem <= max_val:
            idx = val_to_idx.get(rem, -1)
            if idx >= start_idx:
                return True
        
        if rem not in div_to_indices:
            return False
        
        key = (rem, start_idx)
        if key in min_max_val:
            return min_max_val[key] <= max_val
        
        best_mm = None  # minimum max_val that works
        for idx in div_to_indices[rem]:
            if idx < start_idx:
                continue
            f = candidates[idx]
            if f > max_val:
                break
            if f > rem:
                break
            
            new_rem = rem // f
            if can_decompose(new_rem, idx + 1, max_val):
                # This branch works with max_val
                # The effective max for this branch is max(f, max_val) but since f <= max_val, it's max_val
                # But we want to track the minimum max_val that could work
                if best_mm is None or max_val < best_mm:
                    best_mm = max_val
                break  # Found one that works, no need to check more for this call
        
        if best_mm is not None:
            min_max_val[key] = best_mm
            return True
        else:
            # Doesn't work with this max_val, but might work with larger
            # We can't say it never works, so don't memoize as False
            # Actually, if it doesn't work with max_val, it might work with larger max_val
            # So we can't memoize a False for a specific max_val
            # But we can memoize: if it doesn't work with max_val, the minimum that works is > max_val
            # This is tricky. Let's just not memoize False for now.
            return False
    
    def find_solution(rem, start_idx, max_val, path, time_limit):
        if time.time() > time_limit:
            return None
        
        if rem == 1:
            return path[:]
        
        if len(path) >= 12:
            return None
        
        if rem < 13:
            return None
        
        if 12 < rem <= max_val:
            idx = val_to_idx.get(rem, -1)
            if idx >= start_idx:
                return path + [rem]
        
        if rem not in div_to_indices:
            return None
        
        for idx in div_to_indices[rem]:
            if idx < start_idx:
                continue
            f = candidates[idx]
            if f > max_val:
                break
            if f > rem:
                break
            
            new_rem = rem // f
            new_path = path + [f]
            result = find_solution(new_rem, idx + 1, max_val, new_path, time_limit)
            if result is not None:
                return result
        
        return None
    
    # Get an initial solution with a reasonable bound
    # Start with a moderate M, say the 10th percentile of candidates
    # Actually, let's just use target as the bound initially
    sol = find_solution(target, 0, target, [], deadline)
    if sol is None:
        return [target]
    
    best_M = max(sol)
    best_solution = sol
    
    # Now try to find smaller M values
    # Try M values in decreasing order from best_M
    for i in range(len(candidates) - 1, -1, -1):
        if time.time() > deadline:
            break
        M = candidates[i]
        if M >= best_M:
            continue
        if can_decompose(target, 0, M):
            sol = find_solution(target, 0, M, [], deadline)
            if sol is not None:
                best_solution = sol
                best_M = M
                # Keep going to find even smaller M
    
    if best_solution is None:
        return [target]
    
    return best_solution
