import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    target = math.factorial(12)
    
    # 1. Calculate all divisors of target
    divs = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divs.append(i)
            if i != target // i:
                divs.append(target // i)
    divs.sort()
    
    # Filter divisors > 12
    # We only care about divisors in [13, target]
    # But for binary search, we check subsets.
    
    # Function to check if a valid decomposition exists with max factor <= D
    def can_decompose(D):
        # Candidates are divisors d such that 13 <= d <= D
        cands = [d for d in divs if 13 <= d <= D]
        if not cands:
            return None
        
        # Create a set for O(1) lookup of if rem is a valid candidate
        # Note: cands is sorted.
        cand_set = set(cands)
        # Map value to index for quick access? Or just check set and then find index?
        # Since we need to ensure distinctness (index > min_idx), we need to know if rem is in cands[i:]
        # If rem is in cand_set, we find its index. If index > min_idx, it's a valid single step.
        
        solution = None
        
        def dfs(rem, min_idx, path):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            if rem == 1:
                solution = path[:]
                return
            
            # Max 12 factors
            if len(path) >= 12:
                return
            
            # Optimization: If rem itself is a valid candidate (in cands and index > min_idx),
            # we can complete the path immediately.
            if rem in cand_set:
                # Find index of rem in cands
                # Since cands is sorted, we can use bisect
                idx = bisect.bisect_left(cands, rem)
                if idx < len(cands) and cands[idx] == rem and idx > min_idx:
                    solution = path + [rem]
                    return
            
            # Iterate candidates from min_idx onwards
            for i in range(min_idx, len(cands)):
                d = cands[i]
                if d > rem:
                    break
                
                if rem % d == 0:
                    dfs(rem // d, i + 1, path + [d])
                    if solution is not None:
                        return
        
        dfs(target, 0, [])
        return solution

    # Binary search for the minimum D
    low = 13
    high = target
    best_sol = None
    
    # We can also do a linear search from 13 upwards if we expect D to be small.
    # Binary search is safer for large D.
    # However, the DFS complexity depends on D.
    # Small D -> small cands -> fast DFS.
    # Large D -> large cands -> slow DFS.
    # Binary search will test large D first (midpoint).
    # Midpoint is ~2.4e8. cands size ~700.
    # DFS with 700 candidates might be slow.
    
    # Better strategy:
    # 1. Try linear search for small D (13 to 200). This is likely where the answer is.
    # 2. If not found, fallback to binary search or larger linear steps.
    
    # Let's try linear up to 200 first.
    for D in range(13, 201):
        if time.time() > deadline:
            break
        sol = can_decompose(D)
        if sol is not None:
            return sol
            
    # If not found in 13..200, use binary search on [201, target]
    # Or just extend linear search in larger steps?
    # Binary search is good.
    
    low = 201
    high = target
    while low < high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        sol = can_decompose(mid)
        if sol is not None:
            high = mid
            best_sol = sol
        else:
            low = mid + 1
            
    if best_sol is not None:
        return best_sol
        
    # If binary search didn't update best_sol (e.g. timeout or no sol in range?), 
    # check low explicitly?
    # If no solution found, return [target] as fallback (max factor = target)
    # But [target] is always a valid solution (1 factor, >12, distinct).
    # Wait, is [target] valid?
    # Distinct integers: Yes (just one).
    # All > 12: Yes.
    # Product: Yes.
    # Max factor: target.
    
    # Let's make sure we return the best found.
    # If we didn't find anything, [target] is the answer.
    # But we likely found something in 13..200.
    
    return [target]
