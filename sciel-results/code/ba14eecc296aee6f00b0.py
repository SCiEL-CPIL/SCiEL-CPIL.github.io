import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # We want a C4-free graph G on n vertices with min_degree >= n - 39.
    # Equivalently, complement H has max_degree <= 38.
    # Let's try to construct G directly with a greedy approach.
    
    # Strategy: For each n from 47 down to 39, try to find a C4-free graph
    # on n vertices with min_degree >= n-39.
    
    # A C4-free graph: no four vertices a,b,c,d with edges ab,bc,cd,da.
    # To check if adding edge (u,v) creates a C4: check if there exists w such that
    # both u-w and v-w are edges (forming a path u-w-v of length 2), AND there exists
    # x such that u-x and v-x are edges... wait, that's not quite right.
    
    # Adding edge (u,v) creates a C4 if there exists a vertex w such that
    # w is adjacent to both u and v in the current graph, AND there exists another
    # vertex x adjacent to both u and v... no.
    
    # A C4 is u-v-w-x-u. If I add edge (u,v), it creates a C4 if there exist
    # distinct vertices w,x such that edges u-w, w-x, x-v exist in the current graph.
    # Equivalently: there exist w,x with w != x, w != u,v, x != u,v, such that
    # u~w, w~x, x~v all exist.
    
    # So when adding edge (u,v), I need to check: do there exist two distinct vertices
    # w and x (both different from u and v) such that u~w, w~x, x~v?
    
    # This is equivalent to: is there a path of length 2 from u to v through two
    # intermediate vertices? I.e., does there exist w adjacent to u and x adjacent to v
    # with w~x and w != x, w,v,u all distinct, x,v,u all distinct?
    
    # More precisely: for each common neighbor... wait, no. It's a path u-w-x-v where
    # w and x are distinct and neither is u or v.
    
    # Let me define: N(u) = neighbors of u, N(v) = neighbors of v.
    # Adding (u,v) creates a C4 iff there exist w in N(u), x in N(v) with w != x,
    # w not in {u,v}, x not in {u,v}, and w~x.
    
    # Actually, we also need w != v and x != u, but since w in N(u) and the graph is
    # simple, w != u. Similarly x != v. We need w != v and x != u to avoid triangles
    # being miscounted, but actually a triangle u-v-w is not a C4. Let me be careful.
    
    # A C4 requires 4 distinct vertices. So u, v, w, x must all be distinct.
    # w in N(u) means w != u. x in N(v) means x != v.
    # We need w != v (else the path is u-v-w which is just two edges, not a C4)
    # and x != u (else u-x-v is two edges).
    # And w != x.
    
    # So: adding (u,v) creates a C4 iff there exist w, x such that:
    #   w in N(u), x in N(v), w != v, x != u, w != x, and w~x.
    
    # For efficiency, I can precompute for each pair (u,v) whether adding the edge
    # would create a C4.
    
    # Let me implement a greedy algorithm:
    # 1. Start with an empty graph on n vertices.
    # 2. Repeatedly add edges that don't create a C4, prioritizing those that
    #    increase the minimum degree or connect low-degree vertices.
    # 3. Stop when no more edges can be added.
    # 4. Check if min_degree >= n-39.
    
    # But starting from empty might not reach high enough density. Let me start
    # from a known C4-free graph structure.
    
    # Alternative: Use a polarity graph as a starting point, then add edges.
    
    # Actually, let me try a randomized greedy approach with multiple restarts.
    
    def would_create_c4(adj, u, v, n):
        """Check if adding edge (u,v) would create a C4 in the graph with adjacency matrix adj."""
        # Need to find w, x such that:
        # w in N(u), x in N(v), w != v, x != u, w != x, and adj[w][x] == 1
        # For efficiency, iterate over common structure.
        
        # For each w adjacent to u (w != v):
        #   For each x adjacent to v (x != u, x != w):
        #     if adj[w][x]: return True
        # This is O(deg(u) * deg(v)) which could be slow for dense graphs.
        
        # Better: for each w in N(u) \ {v}, check if any neighbor of w (excluding u)
        # is in N(v) \ {u, w}.
        
        # Actually, let's use: for each w in N(u) \ {v}, for each x in N(w) \ {u, v}:
        #   if adj[x][v]: return True
        # This is sum over w in N(u)\{v} of deg(w).
        
        # Or even better: precompute for each pair (u,v) the set of "blocking" vertices.
        
        # Let me just do the direct check for now and optimize if needed.
        for w in range(n):
            if w == u or w == v:
                continue
            if not adj[u][w]:
                continue
            # w is a candidate intermediate
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if adj[w][x] and adj[x][v]:
                    return True
        return False
    
    def try_construct(n, seed, time_limit):
        """Try to construct a C4-free graph on n vertices with min_degree >= n-39."""
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Start with a random C4-free graph by adding random edges
        # in random order, skipping those that create C4s.
        
        # Generate all possible edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(edges)
        
        # Add edges greedily
        for u, v in edges:
            if time.time() > time_limit:
                break
            if not adj[u][v] and not would_create_c4(adj, u, v, n):
                adj[u][v] = True
                adj[v][u] = True
        
        # Check min degree
        degrees = adj.sum(axis=1)
        min_deg = int(degrees.min())
        
        if min_deg >= n - 39:
            # Extract edge list
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append([i, j])
            return {'n': n, 'edges': edge_list}
        
        return None
    
    # Try n from 47 down to 39, with multiple random seeds
    best_result = {'n': 39, 'edges': []}  # fallback
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        # For n=47, need min_deg >= 8
        # For n=46, need min_deg >= 7
        # etc.
        
        # Try multiple random restarts
        max_restarts = 50 if n <= 44 else 20
        
        for seed in range(max_restarts):
            if time.time() > deadline - 10:  # leave 10s margin
                break
            
            result = try_construct(n, seed, deadline - 15)
            if result is not None:
                best_result = result
                # Found a valid graph for this n, return it
                return best_result
        
        # If we didn't find one for this n, continue to lower n
    
    return best_result
