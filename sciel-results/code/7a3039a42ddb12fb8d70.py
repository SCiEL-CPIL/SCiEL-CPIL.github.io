import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = factorial(12)
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
    divisors.sort()
    
    # Pre-compute divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # Iterate M from 13 upwards, find smallest M with a valid decomposition
    for M in range(13, N + 1):
        if time.time() > deadline:
            break
        
        # Candidates are divisors d such that 12 < d <= M
        candidates = [d for d in valid_divisors if d <= M]
        
        if not candidates:
            continue
        
        # Sort descending for faster target reduction
        candidates.sort(reverse=True)
        
        def dfs(target, candidates_list, current_set):
            if target == 1:
                return list(current_set)
            
            if not candidates_list:
                return None
            # Pruning: smallest available candidate
            if target < candidates_list[-1]:
                return None
                
            for i, d in enumerate(candidates_list):
                if d > target:
                    continue
                if target % d != 0:
                    continue
                
                # Additional pruning: after picking d, check if new target is achievable
                # with remaining candidates. Quick check: new target must be >= smallest
                # remaining candidate, or be 1.
                new_target = target // d
                if new_target != 1 and i + 1 < len(candidates_list):
                    if new_target < candidates_list[i+1]:
                        continue
                
                current_set.append(d)
                res = dfs(new_target, candidates_list[i+1:], current_set)
                if res:
                    return res
                current_set.pop()
            return None
        
        solution = dfs(N, candidates, [])
        if solution:
            prod = 1
            for x in solution:
                prod *= x
            if prod == N and len(set(solution)) == len(solution) and all(x > 12 for x in solution) and len(solution) <= 12:
                return solution
    
    return [N]
