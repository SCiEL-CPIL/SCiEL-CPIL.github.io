import time
from math import factorial
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We'll work with the remaining prime exponents
    
    def time_check():
        return time.time() - start_time > DEADLINE
    
    # Generate all divisors of 12! that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Number of divisors = 11 * 6 * 3 * 2 * 2 = 792
    
    def get_divisors():
        divs = [1]
        # Multiply by each prime power
        for p, e in [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]:
            new_divs = []
            for d in divs:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return sorted(divs)
    
    all_divs = get_divisors()
    valid_divs = [d for d in all_divs if d > 12]
    valid_set = set(valid_divs)
    
    # For a given max M, get valid divisors <= M
    def get_cands(M):
        return [d for d in valid_divs if d <= M]
    
    # DFS to find distinct factors with product = target, all in cands, max <= M
    # cands must be sorted
    def dfs(cands, target, max_factors=12):
        n = len(cands)
        # Precompute suffix products for pruning
        # suffix_prod[i] = product of cands[i:] but that's too big
        # Instead, use log for pruning
        
        import math
        log_target = math.log(target)
        
        # For pruning: if we're at index i, we can pick at most (max_factors - len(chosen)) more
        # The max product we can achieve is product of the largest (max_factors - len(chosen)) elements from cands[i:]
        # But that's expensive to compute. Use log.
        
        # Precompute log of cands
        log_cands = [math.log(c) for c in cands]
        
        # Suffix sum of logs for quick max product estimation
        # suffix_log[i] = sum of log_cands[i:]
        # But we can only pick a subset, not all.
        # The max product from cands[i:] picking at most k elements is the product of the largest k elements.
        
        # For simplicity, let's just use a simple bound:
        # If current_prod * (max_cand)^remaining < target, prune
        
        def _search(start_idx, current_prod, chosen, remaining_factors):
            if time_check():
                return None
            
            if current_prod == target:
                if 1 <= len(chosen) <= 12:
                    return list(chosen)
                return None
            
            if current_prod > target:
                return None
            
            if len(chosen) >= 12:
                return None
            
            # Pruning: even if we pick the largest available factors, can we reach target?
            remaining_needed = 12 - len(chosen)
            if remaining_needed == 0:
                return None
            
            # Max possible: pick the largest min(remaining_needed, n - start_idx) from cands[start_idx:]
            avail_count = n - start_idx
            if avail_count == 0:
                return None
            
            pick_count = min(remaining_needed, avail_count)
            # The largest pick_count elements are at the end of cands[start_idx:]
            # i.e., cands[n-pick_count : n]
            max_prod = 1
            for j in range(pick_count):
                max_prod *= cands[n - 1 - j]
            
            if current_prod * max_prod < target:
                return None
            
            # Also: if current_prod > target, already handled
            # Min possible: we need to be able to divide target by current_prod to get an integer
            # And that quotient must be achievable by distinct factors from cands[start_idx:]
            
            remaining_target = target // current_prod
            if current_prod * (target // current_prod) != target:
                # target not divisible by current_prod, but that's fine since we're building up
                # Actually, we need target % current_prod == 0 for the remaining to be integer
                # But current_prod divides target by construction (since we only multiply by divisors)
                pass
            
            # Try each candidate
            for i in range(start_idx, n):
                d = cands[i]
                
                if d > remaining_target:
                    break
                
                if remaining_target % d != 0:
                    continue
                
                new_chosen = chosen + [d]
                res = _search(i + 1, current_prod * d, new_chosen, remaining_factors - 1)
                if res:
                    return res
                
                if time_check():
                    return None
            
            return None
        
        return _search(0, 1, [], 12)
    
    # Binary search on M
    lo = 13
    hi = target
    
    # Quick check: is there a solution with M = target? Yes, [target]
    # But we want to minimize M
    
    best_sol = [target]
    best_M = target
    
    while lo <= hi:
        mid = (lo + hi) // 2
        
        if time_check():
            break
        
        cands = get_cands(mid)
        if len(cands) < 1:
            lo = mid + 1
            continue
        
        sol = dfs(cands, target)
        if sol:
            best_sol = sol
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_sol
