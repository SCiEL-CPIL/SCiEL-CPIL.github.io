import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + divisors[i]
        
    best_count = float('inf')
    best_path = []
    
    def dfs(idx, remaining, count, path):
        nonlocal best_count, best_path
        
        # Base case: No more divisors to consider
        if idx == num_divs:
            if remaining == 0 and count < best_count:
                best_count = count
                best_path = path[:]
            return

        # Pruning 1: Can we form the remaining sum with the rest?
        if remaining > suffix_sum[idx]:
            return

        # Pruning 2: Can we beat the current best count?
        # Max divisor available is divisors[idx]
        min_needed = (remaining + divisors[idx] - 1) // divisors[idx]
        if count + min_needed >= best_count:
            return

        # Option 1: Include divisors[idx]
        if divisors[idx] <= remaining:
            path.append(divisors[idx])
            dfs(idx + 1, remaining - divisors[idx], count + 1, path)
            path.pop()
            
        # Option 2: Exclude divisors[idx]
        dfs(idx + 1, remaining, count, path)

    dfs(0, target, 0, [])
    
    if best_count == float('inf'):
        raise ValueError("No solution found")
        
    return best_path
