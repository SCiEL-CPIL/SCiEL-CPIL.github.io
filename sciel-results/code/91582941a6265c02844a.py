import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Generate all divisors of N
    divisors = set()
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    best_max = float('inf')
    best_factors = []
    
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Greedy initialization to get a quick initial solution
    def greedy_solve():
        remaining = N
        factors = []
        for d in valid_divs:
            if remaining == 1:
                break
            if d > remaining:
                break
            if remaining % d == 0:
                factors.append(d)
                remaining //= d
        if remaining == 1 and len(factors) >= 1:
            return factors
        return None
    
    greedy_result = greedy_solve()
    if greedy_result:
        best_max = max(greedy_result)
        best_factors = greedy_result
    
    # DFS to find a solution with smaller maximum factor
    def dfs(start_idx, current_prod, path):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if current_prod == N:
            if len(path) >= 1:
                max_val = path[-1]  # Since we add in increasing order, last is max
                if max_val < best_max:
                    best_max = max_val
                    best_factors = list(path)
            return
        
        if current_prod > N:
            return
        
        # Find the largest index where valid_divs[i] < best_max
        end_idx = len(valid_divs)
        for j in range(start_idx, len(valid_divs)):
            if valid_divs[j] >= best_max:
                end_idx = j
                break
        
        if end_idx <= start_idx:
            return
        
        for i in range(start_idx, end_idx):
            d = valid_divs[i]
            
            if current_prod * d > N:
                break
            
            if N % (current_prod * d) != 0:
                continue
            
            path.append(d)
            dfs(i + 1, current_prod * d, path)
            path.pop()
    
    dfs(0, 1, [])
    
    if not best_factors:
        return [N]
    
    return best_factors
