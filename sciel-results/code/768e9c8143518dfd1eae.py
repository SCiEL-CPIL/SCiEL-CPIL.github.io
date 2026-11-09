import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030 # LCM of divisors
    
    # Precompute f_i(r) = distance from r to next multiple of d_i
    # f_i(r) = (-r) % d_i
    # If result is 0, distance is 0.
    
    # We will search for the minimum W
    best_W = L # Upper bound
    best_solution = None
    
    # To speed up, we can iterate A.
    # For each A, compute min W needed for at least one multiple of each.
    # Then check matching for W from min_W to best_W - 1.
    
    # Optimization: If we find a solution, we can stop early if we hit a lower bound?
    # The lower bound is at least max(d_i) - 1? No.
    # But we can just iterate A.
    
    # We can also randomize A or just iterate all.
    # L=30030 is small enough to iterate all.
    
    for A in range(L):
        # Calculate minimum W such that each divisor has at least one multiple in [A, A+W]
        min_W = 0
        for i in range(n):
            d = divisors[i]
            r = A % d
            dist = (-r) % d
            if dist > min_W:
                min_W = dist
        
        if min_W >= best_W:
            continue
            
        # Try W from min_W to best_W - 1
        for W in range(min_W, best_W):
            # Check if matching exists in [A, A+W]
            # Generate candidates for each i
            candidates = [[] for _ in range(n)]
            
            feasible = True
            for i in range(n):
                d = divisors[i]
                r = A % d
                dist = (-r) % d
                start = A + dist
                # Candidates are start, start + d, start + 2d, ... <= A + W
                # Also x must be > 0
                x = start
                while x <= A + W:
                    if x > 0:
                        candidates[i].append(x)
                    x += d
                
                if not candidates[i]:
                    feasible = False
                    break
            
            if not feasible:
                continue
                
            # Check bipartite matching
            # Left: 0..n-1, Right: candidates
            # We can map candidates to indices.
            # Since W is small, number of candidates is small.
            
            # For efficiency, we can just use the values as keys in a dict for matching?
            # Or map value to index.
            
            # Let's use a simple DFS matching.
            # match_right: dict value -> left_index
            # Or since values are unique in the window (W < L), we can use a set of used values.
            
            used_values = set()
            
            def dfs(i, visited_values):
                for x in candidates[i]:
                    if x in visited_values:
                        continue
                    visited_values.add(x)
                    if x not in used_values or dfs(used_values[x], visited_values):
                        used_values[x] = i
                        return True
                return False
            
            # Reset used_values for each i in the outer loop of matching
            # Wait, standard Hopcroft-Karp or Kuhn's algorithm:
            # match_right is global for the whole matching attempt.
            
            match_right = {} # value -> i
            
            def dfs_kuhn(i, visited):
                for x in candidates[i]:
                    if x in visited:
                        continue
                    visited.add(x)
                    if x not in match_right or dfs_kuhn(match_right[x], visited):
                        match_right[x] = i
                        return True
                return False
            
            found = True
            for i in range(n):
                visited = set()
                if not dfs_kuhn(i, visited):
                    found = False
                    break
                    
            if found:
                # Construct result
                result = [0] * n
                for x, i in match_right.items():
                    result[i] = x
                    
                # Verify
                if len(set(result)) == n and all(x > 0 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                    best_W = W
                    best_solution = result
                    break # No need to check larger W for this A
                    
    if best_solution is None:
        # Fallback
        return divisors[:]
        
    return best_solution
