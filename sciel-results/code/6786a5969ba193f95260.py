import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Minimum width required for n distinct integers is n-1
    # So we start W from n-1
    min_w = n - 1
    
    # We will search for the minimum W
    # For each W, we check if there exists an L such that 
    # there is a perfect matching between divisors and integers in [L, L+W]
    
    # To ensure we don't miss a solution, we should check L up to a reasonable bound.
    # Checking L up to 5000 is safe and fast.
    
    max_L_check = 5000
    
    # Helper: Kuhn's algorithm for bipartite matching
    def get_matching(adj):
        """
        adj: list of lists. adj[i] contains indices of right nodes connected to left node i.
        Returns match_to array where match_to[u] = v if right node u is matched to left node v, 
        or None if no perfect matching exists.
        """
        n_left = len(adj)
        
        # Determine size of right side
        max_u = 0
        for i in range(n_left):
            if adj[i]:
                max_u = max(max_u, max(adj[i]))
        
        if max_u == 0 and all(len(a) == 0 for a in adj):
             return None
             
        match_to = [-1] * (max_u + 1)
        
        def try_kuhn(v, used):
            if used[v]:
                return False
            used[v] = True
            for u in adj[v]:
                if match_to[u] == -1 or try_kuhn(match_to[u], used):
                    match_to[u] = v
                    return True
            return False
            
        # We need to match all left nodes.
        for v in range(n_left):
            used = [False] * n_left
            if not try_kuhn(v, used):
                return None
        return match_to

    # Iterate W
    for W in range(min_w, 100): # Cap W at 100, should find solution well before
        if time.time() > deadline:
            break
            
        # Iterate L
        for L in range(1, max_L_check + 1):
            if time.time() > deadline:
                break
                
            # Build adjacency list
            # Left nodes: 0..n-1 (divisors)
            # Right nodes: 0..W (offsets in interval)
            # Integer value is L + offset
            # Divisor d_i is matched to offset j if (L+j) % d_i == 0
            
            adj = [[] for _ in range(n)]
            valid = True
            
            for i, d in enumerate(divisors):
                for j in range(W + 1):
                    val = L + j
                    if val % d == 0:
                        adj[i].append(j)
                if not adj[i]:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            match_to = get_matching(adj)
            
            if match_to is not None:
                # Reconstruct solution
                result = []
                for i in range(n):
                    # Find which right node u is matched to left node i
                    # match_to[u] == i
                    found = False
                    for u in range(len(match_to)):
                        if match_to[u] == i:
                            result.append(L + u)
                            found = True
                            break
                    if not found:
                        # Should not happen if perfect matching
                        break
                if len(result) == n:
                    # Verify distinctness and constraints
                    if len(set(result)) == n and all(r % d == 0 for r, d in zip(result, divisors)):
                        return [int(x) for x in result]
                            
    # Fallback if not found (should not happen)
    # Greedy assignment
    result = []
    for d in divisors:
        # Pick smallest multiple not in result
        k = 1
        while True:
            val = k * d
            if val not in result:
                result.append(val)
                break
            k += 1
    return [int(x) for x in result]
