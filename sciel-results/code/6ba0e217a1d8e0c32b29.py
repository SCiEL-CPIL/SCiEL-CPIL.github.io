import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Helper to find a valid assignment for a given L and span
    def find_assignment(L, span):
        # 1. Generate candidates and check existence
        candidates = []
        for d in divisors:
            k = (L + d - 1) // d
            val = k * d
            cand_list = []
            while val <= L + span:
                cand_list.append(val)
                val += d
            if not cand_list:
                return None
            candidates.append(cand_list)
        
        # 2. Sort indices by number of candidates (ascending) for better pruning
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        assignment = [0] * n
        used = set()
        
        # Iterative DFS to avoid recursion overhead/limits? 
        # Recursion depth is only 10, so it's fine.
        def dfs(idx):
            if idx == n:
                return True
            orig_idx = indices[idx]
            for v in candidates[orig_idx]:
                if v not in used:
                    used.add(v)
                    assignment[orig_idx] = v
                    if dfs(idx + 1):
                        return True
                    used.remove(v)
                    assignment[orig_idx] = 0
            return False
            
        if dfs(0):
            return assignment
        return None

    # Iterate span from minimum possible (9) upwards
    # 10 distinct integers require a span of at least 9.
    for span in range(9, LCM + 1):
        if time.time() > deadline:
            break
            
        for L in range(1, LCM + 1):
            if time.time() > deadline:
                break
                
            # Quick check: Does each divisor have at least one multiple in [L, L+span]?
            valid = True
            for d in divisors:
                m = ((L + d - 1) // d) * d
                if m > L + span:
                    valid = False
                    break
            if not valid:
                continue
                
            res = find_assignment(L, span)
            if res is not None:
                return [int(x) for x in res]
                
    # Fallback: Should not be reached if LCM is large enough
    # But to satisfy type checks and return something, we return a known valid solution 
    # or let it fail. The problem guarantees a solution exists.
    # If we hit deadline, we might return an invalid result. 
    # To be safe, if we exit loops without finding, we should return a valid static solution if we have one, 
    # but we don't know one. 
    # However, the search space is small enough that we should find it.
    
    # If we get here, it means we didn't find a solution in time.
    # This would result in an invalid return (zeros).
    # We can try to return a greedy solution as a last resort?
    # No, better to ensure the search is efficient enough.
    
    return [0] * n
