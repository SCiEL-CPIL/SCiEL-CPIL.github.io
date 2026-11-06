import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate Paley graph on 29 vertices
    q = 29
    n = q
    squares = {x * x % q for x in range(1, q)}
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in squares:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
    
    # Helper to check validity
    def is_valid(adj, n):
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]: continue
                for k in range(j + 1, n):
                    if not adj[i][k] or not adj[j][k]: continue
                    for l in range(k + 1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return False
        # Check Independent Set of 6
        # Heuristic check for small n, exact for n <= 30?
        # For n=29, exact check is 29 choose 6 ~ 475k. Fast.
        for combo in itertools.combinations(range(n), 6):
            is_independent = True
            for u in combo:
                for v in combo:
                    if u < v and adj[u][v]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        return True

    # Verify base graph
    if not is_valid(adj, n):
        # Fallback to P17 if P29 fails (it shouldn't)
        q = 17
        n = q
        squares = {x * x % q for x in range(1, q)}
        adj = [[False] * n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in squares:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append([i, j])
        
    # Try to extend
    current_n = n
    for new_v in range(current_n, 35):
        if time.time() > deadline - 5.0:
            break
        
        # Get triangles and independent sets of size 5 in current graph
        triangles = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if not adj[i][j]: continue
                for k in range(j + 1, current_n):
                    if adj[i][k] and adj[j][k]:
                        triangles.append((i, j, k))
        
        indep_sets_5 = []
        for combo in itertools.combinations(range(current_n), 5):
            is_indep = True
            for u in combo:
                for v in combo:
                    if u < v and adj[u][v]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                indep_sets_5.append(combo)
        
        # Find valid neighbor set S for new_v
        # S must be triangle-free in G and hit all indep_sets_5
        found = False
        # Try random subsets? Or greedy?
        # Greedy: Start with S = all vertices.
        # Remove vertices that cause triangles?
        # This is hard.
        
        # Try a simple heuristic:
        # Randomly pick a vertex to be in S or not?
        # Or try S = vertices with low degree?
        
        # Let's try a limited search for S
        # S is a subset of 0..current_n-1
        # Constraint 1: No triangle in G[S]
        # Constraint 2: S intersects every set in indep_sets_5
        
        # If indep_sets_5 is empty, S can be anything triangle-free.
        # If triangles is empty, S can be anything hitting indep_sets_5.
        
        # Try S = all vertices
        # If G[all] has triangles, remove one vertex from each triangle?
        
        # Simple approach:
        # Try S = random subset of size ~ current_n/2
        # Check constraints
        
        # Better: 
        # Since we just want SOME valid S, we can try to construct one.
        # If we fail after some attempts, we break.
        
        attempts = 0
        while not found and attempts < 100 and time.time() < deadline - 5.0:
            attempts += 1
            # Random S
            S = set()
            for v in range(current_n):
                if random.random() < 0.5:
                    S.add(v)
            
            # Check Triangle Free
            valid_tri = True
            for t in triangles:
                if t[0] in S and t[1] in S and t[2] in S:
                    valid_tri = False
                    break
            if not valid_tri:
                continue
                
            # Check Hitting Set
            valid_hit = True
            for i_set in indep_sets_5:
                if not any(v in S for v in i_set):
                    valid_hit = False
                    break
            if not valid_hit:
                continue
                
            # Valid S found
            found = True
            
        if found:
            # Add new vertex
            new_idx = current_n
            # Extend adj
            adj.append([False] * (current_n + 1))
            for i in range(current_n):
                adj[i].append(False)
            for v in S:
                adj[new_idx][v] = True
                adj[v][new_idx] = True
                edges.append([new_idx, v])
            current_n += 1
        else:
            break
            
    return {'n': current_n, 'edges': edges}
