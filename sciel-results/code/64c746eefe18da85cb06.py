import time
from math import factorial

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    target = factorial(12)  # 479001600
    
    def time_check():
        return time.time() - start_time > DEADLINE
    
    def get_divisors():
        divs = [1]
        for p, e in [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]:
            new_divs = []
            for d in divs:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return sorted(divs)
    
    all_divs = get_divisors()
    valid_divs = [d for d in all_divs if d > 12]
    
    def get_cands(M):
        return [d for d in valid_divs if d <= M]
    
    def dfs(cands, target):
        n = len(cands)
        if n == 0:
            return None
        
        def _search(start_idx, current_prod, chosen):
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
            
            remaining_target = target // current_prod
            if current_prod * (target // current_prod) != target:
                return None
                
            # Pruning: Max possible product from remaining candidates
            remaining_slots = 12 - len(chosen)
            avail_count = n - start_idx
            if avail_count == 0:
                return None
                
            pick_count = min(remaining_slots, avail_count)
            if pick_count == 0:
                return None
            
            max_prod = 1
            # Largest elements are at the end of cands
            # We need to pick distinct elements from cands[start_idx:]
            # The max product is achieved by picking the largest available elements.
            # Since we must pick in increasing order (to ensure distinctness and avoid permutations in this DFS structure),
            # actually we are just picking a subset. The order of picking doesn't change the set.
            # But the DFS enforces index increasing, which means values increasing (since cands is sorted).
            # So we are picking a strictly increasing sequence.
            # The max product of any subset of size k from cands[start_idx:] is the product of the largest k elements.
            for j in range(pick_count):
                max_prod *= cands[n - 1 - j]
                
            if current_prod * max_prod < target:
                return None
                
            # Also, if the smallest possible product (picking smallest available) exceeds remaining_target?
            # No, we just need *some* product to equal remaining_target.
            
            for i in range(start_idx, n):
                d = cands[i]
                
                if d > remaining_target:
                    break
                
                if remaining_target % d != 0:
                    continue
                
                chosen.append(d)
                res = _search(i + 1, current_prod * d, chosen)
                if res:
                    return res
                chosen.pop()
                
                if time_check():
                    return None
            
            return None
        
        return _search(0, 1, [])
    
    lo = 13
    hi = target
    
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
