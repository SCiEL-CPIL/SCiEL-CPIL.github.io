import math

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12 and sort
    divisors = sorted([d for d in divisors if d > 12])
    
    best_max = float('inf')
    best_factors = None
    
    # To speed up finding a good initial solution, we can try a greedy approach
    # or just let the DFS handle it. 
    # However, to ensure we prune effectively, having a finite best_max early is good.
    # Let's just run the DFS.
    
    def dfs(start_idx, current_prod, chosen, current_max):
        nonlocal best_max, best_factors
        
        if current_prod == N:
            if current_max < best_max:
                best_max = current_max
                best_factors = chosen[:]
            return
        
        if current_prod > N:
            return
            
        # Prune: if the current max is already not better than the best found, stop
        if current_max >= best_max:
            return
            
        # Prune: if even multiplying by the largest possible remaining factors 
        # cannot reach N, we might still be able to reach it with MORE factors?
        # No, if we are at index i, we can pick any divisor from i onwards.
        # The product of ALL remaining divisors is huge, so this bound is weak.
        # But we can check: if current_prod * (max divisor) < N, we can still pick more.
        # So this specific bound is not useful for "can we reach N".
        
        # However, we can bound the *number* of factors? No.
        
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            
            # If this divisor is already too big, break
            if d >= best_max:
                break
                
            if current_prod * d > N:
                break
                
            # Recurse
            dfs(i + 1, current_prod * d, chosen + [d], max(current_max, d))
            
    dfs(0, 1, [], 1)
    
    if best_factors is None:
        return [N]
        
    return best_factors
