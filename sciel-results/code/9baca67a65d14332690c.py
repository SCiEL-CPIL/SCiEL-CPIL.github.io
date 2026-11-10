import math

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    
    # Binary search for minimum M
    low = 13
    high = target
    best_solution = None
    best_M = target
    
    # To speed up, we can iterate M upwards from a lower bound if we have a good guess,
    # but binary search is robust.
    # However, the property "if solution exists for M, it exists for M+1" is TRUE.
    # So binary search works.
    
    for _ in range(60): # Enough for log2(4.7e8) ~ 29
        mid = (low + high) // 2
        candidates = [d for d in valid_divisors if d <= mid]
        candidates.sort(reverse=True)
        
        # DFS check
        memo = {}
        solution_path = []
        
        def dfs(rem, i):
            if rem == 1:
                return True
            if i >= len(candidates):
                return False
            
            state = (rem, i)
            if state in memo:
                return memo[state]
            
            found = False
            for j in range(i, len(candidates)):
                c = candidates[j]
                if c > rem:
                    continue
                if rem % c == 0:
                    # Try this candidate
                    solution_path.append(c)
                    if dfs(rem // c, j + 1):
                        found = True
                        break
                    solution_path.pop()
            
            memo[state] = found
            return found
        
        if dfs(target, 0):
            best_M = mid
            best_solution = solution_path[:]
            high = mid - 1
        else:
            low = mid + 1
            
    return best_solution
