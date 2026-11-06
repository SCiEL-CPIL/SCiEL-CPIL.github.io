"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_n = 39
    best_edges = []
    
    def check_c4(adj, n):
        """Check if graph has a C4 using the common-neighbor characterization.
        G is C4-free iff for every pair of vertices u,v, the number of common neighbors
        with an edge between them is at most... actually, let's use:
        G has a C4 iff there exist non-adjacent u,v with |N(u)∩N(v)| >= 2 where
        the intersection contains an edge.
        
        More efficient: G has C4 iff there exist u,v (any) such that the number of
        4-cycles through the pair... 
        
        Let me use: G has a C4 iff there exist distinct a,b,c,d with edges a-b, b-c, c-d, d-a.
        Equivalently: there exist non-adjacent u,v such that N(u)∩N(v) contains an edge.
        """
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    # Check if N(u) ∩ N(v) has an edge
                    common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                    if len(common) >= 2:
                        for i in range(len(common)):
                            for j in range(i+1, len(common)):
                                if adj[common[i]][common[j]]:
                                    return True
        return False
    
    def try_construct(n, max_attempts=20, time_limit=None):
        """Try to construct a C4-free graph on n vertices with min degree >= n-39."""
        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            return True, []
        
        for attempt in range(max_attempts):
            if time_limit and time.time() > time_limit:
                return False, None
            
            adj = [[False] * n for _ in range(n)]
            degrees = [0] * n
            
            # Greedy: add edges that don't create C4, prioritizing low-degree vertices
            # Use a priority queue approach: repeatedly pick the edge that most helps min degree
            
            # Potential edges sorted by a heuristic
            potential = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(potential)
            
            for u, v in potential:
                if time_limit and time.time() > time_limit:
                    return False, None
                
                # Check if adding (u,v) creates a C4
                # A C4 is created iff there exist a,b such that u-a, a-b, b-v are edges
                # i.e., there's an edge (a,b) with a in N(u) and b in N(v)
                
                # Efficient check: iterate over neighbors of u, for each neighbor a,
                # check if any neighbor b of v (with b != a, b != u, b != v) has a-b edge
                # and a != v, b != u
                
                creates_c4 = False
                for a in range(n):
                    if a == u or a == v or not adj[u][a]:
                        continue
                    for b in range(n):
                        if b == u or b == v or b == a or not adj[v][b]:
                            continue
                        if adj[a][b]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                
                if not creates_c4:
                    adj[u][v] = adj[v][u] = True
                    degrees[u] += 1
                    degrees[v] += 1
            
            min_deg = min(degrees) if n > 0 else 0
            if min_deg >= min_deg_needed:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u][v]:
                            edges.append([u, v])
                return True, edges
        
        return False, None
    
    # Try from n=47 down to n=40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        success, edges = try_construct(target_n, max_attempts=30, time_limit=deadline)
        if success:
            best_n = target_n
            best_edges = edges
            break
    
    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
