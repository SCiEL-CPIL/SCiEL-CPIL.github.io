import sys
sys.setrecursionlimit(10000)

def construct():
    n = 16
    # Construct Clebsch graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                edges.append([u, v])
    
    # Represent graph as bitmasks for adjacency
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    # Identify edges already in Clebsch
    clebsch_set = set(map(tuple, edges))
    
    # Identify candidate edges (not in Clebsch)
    candidates = [e for e in all_edges if e not in clebsch_set]
    
    # Filter out edges that form a triangle with two edges in Clebsch
    # If (u,v) is candidate, check if there is w such that (u,w) in Clebsch and (v,w) in Clebsch
    valid_candidates = []
    for u, v in candidates:
        common = adj[u] & adj[v]
        # If common is not 0, there is a w connected to both u and v in Clebsch
        # Adding (u,v) would create triangle (u,v,w)
        if common == 0:
            valid_candidates.append((u, v))
            
    # Now we want to select the maximum number of edges from valid_candidates
    # such that the resulting graph is triangle-free.
    # This is a Maximum Independent Set problem on a conflict graph?
    # Or just a backtracking search.
    
    # To optimize, we can use a bitset representation for the selected edges
    # and check conflicts.
    
    # Let's use a simple backtracking with pruning.
    # Sort candidates by degree in the "conflict graph" or just random?
    # A good heuristic is to try edges that conflict with fewer other candidates.
    
    # Precompute conflicts between valid_candidates
    # Conflict if they form a triangle with each other or with Clebsch?
    # We already filtered out those conflicting with Clebsch (2 edges from Clebsch).
    # But they might conflict with 1 edge from Clebsch and 1 from S?
    # Or 2 from S?
    
    # Let's just do a recursive search on valid_candidates.
    # State: index in valid_candidates, current adj bitmasks, count of added edges.
    
    best_count = 0
    best_edges = []
    
    # To speed up, we can use a bitset for available candidates
    # But n=16 is small, valid_candidates is at most 80.
    
    def backtrack(idx, adj_current, count, current_edges):
        nonlocal best_count, best_edges
        
        if count > best_count:
            best_count = count
            best_edges = current_edges[:]
            
        if idx == len(valid_candidates):
            return
            
        # Upper bound: count + remaining candidates
        if count + (len(valid_candidates) - idx) <= best_count:
            return
            
        # Option 1: Skip valid_candidates[idx]
        backtrack(idx + 1, adj_current, count, current_edges)
        
        # Option 2: Take valid_candidates[idx]
        u, v = valid_candidates[idx]
        
        # Check if adding (u,v) creates a triangle with existing edges in adj_current
        # Triangle if common neighbor w exists
        if (adj_current[u] & adj_current[v]) == 0:
            # Safe to add
            new_adj_u = adj_current[u] | (1 << v)
            new_adj_v = adj_current[v] | (1 << u)
            
            # Copy adj_current (list of ints)
            # This is expensive. We can use a stack of changes or just copy since n=16.
            adj_new = adj_current[:]
            adj_new[u] = new_adj_u
            adj_new[v] = new_adj_v
            
            current_edges.append([u, v])
            backtrack(idx + 1, adj_new, count + 1, current_edges)
            current_edges.pop()

    # Initial adj is Clebsch adj
    backtrack(0, adj, 0, [])
    
    # Combine Clebsch edges and best additional edges
    final_edges = edges + best_edges
    return final_edges
