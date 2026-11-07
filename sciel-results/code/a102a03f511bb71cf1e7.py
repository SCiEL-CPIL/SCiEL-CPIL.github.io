import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    def has_matching(W, m):
        # Determine valid range for x
        start = max(m, 1)
        end = m + W
        if start > end:
            return None
        
        # candidates[i] is list of x in [start, end] divisible by divisors[i]
        # We need to assign distinct x to each i.
        # Build adjacency: left i -> right x
        # Right nodes are x values.
        # Map x to index 0..W for matching?
        # Or just use the values directly.
        
        # Let's use a standard bipartite matching on indices.
        # Left: 0..n-1
        # Right: 0..W (representing start+k)
        
        adj = [[] for _ in range(n)]
        
        # Precompute which x values are in the window
        # x = start + k, k in 0..W
        # Check divisibility
        
        # To speed up, we can iterate k and check which divisors divide start+k
        # Or iterate divisors and find multiples.
        
        # Since W is small, iterating k is good.
        # For each k, x = start + k.
        # For each i, if x % divisors[i] == 0, add edge i -> k.
        
        for k in range(W + 1):
            x = start + k
            for i in range(n):
                if x % divisors[i] == 0:
                    adj[i].append(k)
        
        # Check if any i has no edges
        for i in range(n):
            if not adj[i]:
                return None
        
        # Bipartite matching (Kuhn's algorithm)
        match_to_right = [-1] * (W + 1) # match_to_right[k] = i
        
        def dfs(i, visited):
            for k in adj[i]:
                if visited[k]:
                    continue
                visited[k] = True
                if match_to_right[k] == -1 or dfs(match_to_right[k], visited):
                    match_to_right[k] = i
                    return True
            return False
        
        for i in range(n):
            visited = [False] * (W + 1)
            if not dfs(i, visited):
                return None
        
        # Reconstruct solution
        # match_to_right[k] = i means k is assigned to i
        # We need x_i
        res = [0] * n
        for k in range(W + 1):
            if match_to_right[k] != -1:
                i = match_to_right[k]
                res[i] = start + k
        return res

    start_time = time.time()
    deadline = start_time + 55.0
    
    for W in range(9, 100):
        if time.time() > deadline:
            break
            
        found = False
        for m in range(LCM):
            if time.time() > deadline:
                break
                
            res = has_matching(W, m)
            if res is not None:
                # Check if all positive
                # has_matching ensures start >= 1, so res >= 1
                # Check distinctness (guaranteed by matching)
                # Check divisibility (guaranteed by construction)
                # Check <= 1000000
                if max(res) <= 1000000:
                    return [int(x) for x in res]
                else:
                    # If too large, we might need to shift down?
                    # But m is in [0, LCM). x <= m+W < LCM+W < 30130.
                    # So this check is always true for small W.
                    pass
                    
        if found:
            break

    # Fallback
    reps = [int((i + 1) * LCM) for i in range(n)]
    return reps
