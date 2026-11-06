import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds buffer
    
    # Try to construct for n=47, then 46, etc.
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg_required = n - 39
        if min_deg_required < 1:
            min_deg_required = 1
            
        # Bipartite construction
        # Split into two sets A and B
        n_a = n // 2
        n_b = n - n_a
        
        # We want a bipartite graph with min degree >= min_deg_required
        # and no C4.
        
        # Greedy construction
        # adjacency list
        adj = {i: set() for i in range(n)}
        
        # degrees
        deg = [0] * n
        
        # We need to add edges.
        # To avoid C4, for any edge (u, v), u and v must have no common neighbors.
        # In a bipartite graph, u in A, v in B.
        # Common neighbors of u and v would be in A \cap B, which is empty.
        # Wait. A C4 in a bipartite graph is u1-v1-u2-v2-u1.
        # u1, u2 in A, v1, v2 in B.
        # u1 and u2 share neighbors v1 and v2.
        # So we must ensure that no two vertices in A share more than 1 neighbor in B.
        # And no two vertices in B share more than 1 neighbor in A.
        
        # Greedy:
        # While there is a vertex with degree < min_deg_required:
        #   Pick u with lowest degree.
        #   Pick v in other part with lowest degree such that |N(u) \cap N(v)| <= 0? 
        #   No, if we add (u,v), we create a C4 if there exists u' in A, v' in B 
        #   such that u' is neighbor of v and v' is neighbor of u?
        #   Actually, adding (u,v) creates a C4 if there is a path of length 3 between u and v already.
        #   i.e., u-w-x-v.
        #   In bipartite, u in A, v in B.
        #   Path u-w-x-v means w in B, x in A.
        #   So u is connected to w, w to x, x to v.
        #   This means w is a common neighbor of u and x? No.
        #   w is neighbor of u and x.
        #   x is neighbor of w and v.
        #   So if we add (u,v), we form a C4 if there exists w in B such that u-w and w-x and x-v exist for some x in A.
        #   Basically, if u and v have a common neighbor? No, that's a path of length 2.
        #   C4 is length 4.
        #   So we need to check if there is a path of length 3 between u and v.
        
        # Efficient check:
        # When adding (u,v), check all neighbors w of u (in B).
        # For each w, check all neighbors x of w (in A).
        # If x is a neighbor of v, then we have a C4.
        
        success = False
        for attempt in range(100): # Try a few random orders or just greedy
            if time.time() > deadline:
                break
                
            adj = {i: set() for i in range(n)}
            deg = [0] * n
            
            # Order vertices by index, but maybe shuffle?
            # Greedy with priority on low degree
            while True:
                if time.time() > deadline:
                    break
                
                # Find vertex with min degree
                min_deg = min(deg)
                if min_deg >= min_deg_required:
                    success = True
                    break
                    
                # Pick a vertex u with min degree
                # To be deterministic, pick the first one
                u = 0
                for i in range(n):
                    if deg[i] == min_deg:
                        u = i
                        break
                        
                # Determine part of u
                part_u = 0 if u < n_a else 1
                part_v = 1 - part_u
                
                # Find v in other part
                # Candidates: all vertices in other part
                # Sort candidates by degree
                candidates = []
                if part_v == 0:
                    candidates = list(range(0, n_a))
                else:
                    candidates = list(range(n_a, n))
                    
                candidates.sort(key=lambda x: deg[x])
                
                added = False
                for v in candidates:
                    if deg[v] >= min_deg_required and len(adj[u]) >= min_deg_required:
                        # We only need to add edges if u or v needs more degree?
                        # Actually, we can add edges even if v is full, as long as u needs it.
                        # But if v is full, we can't add more edges to v? 
                        # We can, we just don't need to.
                        # But we must ensure u gets its degree.
                        pass
                    
                    # Check if adding (u,v) creates a C4
                    # Path u-w-x-v
                    # w in N(u), x in N(w)
                    # Check if x in N(v)
                    safe = True
                    for w in adj[u]:
                        for x in adj[w]:
                            if v in adj[x]:
                                safe = False
                                break
                        if not safe:
                            break
                            
                    if safe:
                        # Add edge
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        added = True
                        break
                        
                if not added:
                    # Stuck
                    break
                    
            if success:
                # Verify
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                
                # Double check min degree
                min_d = min(deg)
                if min_d >= min_deg_required:
                    # Double check C4
                    # Fast check: for each pair of edges (u,v) and (u',v'), check if they form C4
                    # Or just trust the greedy check if implemented correctly.
                    # Let's do a quick random verification
                    import itertools
                    has_c4 = False
                    # Check all pairs of vertices in A
                    for u1, u2 in itertools.combinations(range(n_a), 2):
                        common = adj[u1] & adj[u2]
                        if len(common) > 1:
                            has_c4 = True
                            break
                    if not has_c4:
                        return {'n': n, 'edges': edges}
                        
    # Fallback: return the trivial solution
    return {'n': 39, 'edges': []}
