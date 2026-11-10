import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Try to find a solution with small k
    for k in range(8, 15):
        if time.time() > deadline:
            break
        solution = _find_solution(k, deadline)
        if solution:
            return solution
            
    # Fallback to seed logic
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")

def _find_solution(k, deadline):
    def backtrack(start, current_sum, current_count, path):
        if time.time() > deadline:
            return None
        if current_count == k:
            if abs(current_sum - 1.0) < 1e-9:
                return path
            return None
        
        remaining = k - current_count
        target_sum = 1.0 - current_sum
        
        for d in range(start, 10000000):
            if 1/d > target_sum:
                continue
            
            # Pruning: max sum of remaining terms
            # If we pick d+1, d+2, ..., d+remaining-1
            # This is an upper bound on the sum we can add
            max_add = sum(1/(d + i + 1) for i in range(remaining - 1))
            if max_add < target_sum - 1/d:
                continue
                
            res = backtrack(d + 1, current_sum + 1/d, current_count + 1, path + [d])
            if res:
                return res
        return None

    return backtrack(5, 0.0, 0, [])
