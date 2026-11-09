import random
import time
import sys

def construct():
    n = 20
    # Adjacency matrix for fast lookup
    adj = [[0]*n for _ in range(n)]
    
    def is_k33_free():
        # Check if current graph is K3,3 free
        # A K3,3 exists if there are 3 vertices u,v,w sharing 3 common neighbors
        # Or more generally, check all pairs of triangles? No, K3,3 is bipartite.
        # Efficient check: For every triple of vertices (x,y,z), count common neighbors.
        # If any triple has >= 3 common neighbors, it's not K3,3 free.
        # Wait, if x,y,z have common neighbors a,b,c, then {x,y,z} and {a,b,c} form K3,3.
        # So we just need to ensure that for all triples {x,y,z}, |N(x) ∩ N(y) ∩ N(z)| <= 2.
        # Number of triples is C(20,3) = 1140.
        # For each triple, intersection of 3 bitsets is fast.
        return True # Placeholder, we will maintain validity incrementally or check on the fly
        
    def has_k33():
        # Returns True if K3,3 exists
        # Using bitmasks for neighborhoods
        # Precompute bitmasks for each vertex
        masks = [0]*n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
            
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = masks[i] & masks[j] & masks[k]
                    # Count bits
                    if bin(common).count('1') >= 3:
                        return True
        return False

    def try_add_edge(u, v):
        if adj[u][v]:
            return False
        # Check if adding (u,v) creates a K3,3
        # Case 1: u, v on same side. Need w such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
        # Note: u and v are not connected yet, so u not in N(v) and v not in N(u).
        # Common neighbors of u and v: C_uv = N(u) ∩ N(v).
        # If there is a w such that N(w) contains 3 vertices from C_uv, then K3,3.
        # Actually, if we add (u,v), u and v become adjacent.
        # The condition for K3,3 involving edge (u,v) is:
        # 1. u, v same side: Exists w != u,v such that |N(u) ∩ N(v) ∩ N(w)| >= 3.
        #    Note: Since u,v not connected, N(u) and N(v) don't contain each other.
        # 2. u, v opposite sides: Exists v1, v2 in N(u) and u1, u2 in N(v) 
        #    such that u1, u2 are connected to v1, v2.
        #    i.e., N(v1) ∩ N(v2) contains u1 and u2.
        
        # Optimization: Use bitmasks
        mask_u = 0
        mask_v = 0
        for i in range(n):
            if adj[u][i]: mask_u |= (1 << i)
            if adj[v][i]: mask_v |= (1 << i)
            
        # Case 1: u, v same side
        # We need to check if there exists w such that mask_u & mask_v & mask_w has >= 3 bits.
        # But w can be any vertex.
        # Let C_uv = mask_u & mask_v.
        # We need a w such that mask_w & C_uv has >= 3 bits.
        # This is equivalent to: Is there a subset of 3 vertices in C_uv that are all connected to some w?
        # Or simply: For each w, check popcount(mask_w & C_uv) >= 3.
        # This is O(n) per check.
        
        # Case 2: u, v opposite sides
        # We need v1, v2 in N(u) (so bits in mask_u) and u1, u2 in N(v) (bits in mask_v).
        # Such that u1, u2 are connected to v1 and v2.
        # i.e., for some pair v1, v2 in mask_u, and pair u1, u2 in mask_v:
        # adj[v1][u1] and adj[v1][u2] and adj[v2][u1] and adj[v2][u2] are True.
        # This is checking if the bipartite subgraph between N(u) and N(v) contains a K2,2.
        # A K2,2 in the bipartite graph (N(u), N(v)) means two vertices in N(u) share two neighbors in N(v).
        
        # Let's implement the check
        C_uv = mask_u & mask_v
        
        # Case 1 check
        for w in range(n):
            if w == u or w == v: continue
            mask_w = 0
            for i in range(n):
                if adj[w][i]: mask_w |= (1 << i)
            if bin(mask_w & C_uv).count('1') >= 3:
                return False

        # Case 2 check: K2,2 in bipartite graph between N(u) and N(v)
        # Vertices in N(u): list_u
        # Vertices in N(v): list_v
        list_u = [i for i in range(n) if adj[u][i]]
        list_v = [i for i in range(n) if adj[v][i]]
        
        # For each pair in list_v, check if they have 2 common neighbors in list_u
        # Common neighbors of v1, v2 in list_u:
        # N(v1) ∩ N(v2) ∩ list_u
        for idx1 in range(len(list_v)):
            v1 = list_v[idx1]
            mask_v1 = 0
            for i in range(n):
                if adj[v1][i]: mask_v1 |= (1 << i)
            for idx2 in range(idx1+1, len(list_v)):
                v2 = list_v[idx2]
                mask_v2 = 0
                for i in range(n):
                    if adj[v2][i]: mask_v2 |= (1 << i)
                
                common = mask_v1 & mask_v2
                # Check how many bits are in list_u
                # list_u is a set of indices
                count = 0
                for u1 in list_u:
                    if common & (1 << u1):
                        count += 1
                        if count >= 2:
                            return False
                            
        return True

    # Initialize random graph
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.4:
                adj[i][j] = 1
                adj[j][i] = 1

    # Remove edges that create K3,3
    # We can just run a cleanup loop
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Try to remove
                    adj[i][j] = 0
                    adj[j][i] = 0
                    # Check if graph is still K3,3 free? No, removing edges always keeps it free if it was free.
                    # But we want to KEEP edges.
                    # So we should only add edges if they don't break freeness.
                    # This initialization is bad. Let's start empty and add.
                    adj[i][j] = 1
                    adj[j][i] = 1
                    # Actually, let's restart with empty graph and greedy add.
                    break
            if changed: break

    # Restart: Empty graph
    for i in range(n):
        for j in range(n):
            adj[i][j] = 0

    # Greedy Add
    for i in range(n):
        for j in range(i+1, n):
            if try_add_edge(i, j):
                adj[i][j] = 1
                adj[j][i] = 1

    # Local Search
    start_time = time.time()
    while time.time() - start_time < 110:
        improved = False
        # Try adding edges
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    if try_add_edge(i, j):
                        adj[i][j] = 1
                        adj[j][i] = 1
                        improved = True
        # Try removing edges to allow adding others? (2-opt style)
        # For each edge (i,j), remove it, then try to add 2 new edges?
        # Simpler: Try swapping. Remove (i,j), add (k,l) if valid.
        # But adding is usually enough if we start empty.
        
        if not improved:
             # Try more aggressive moves: Remove 1, Add 2
             for i in range(n):
                 for j in range(i+1, n):
                     if adj[i][j]:
                         # Remove
                         adj[i][j] = 0
                         adj[j][i] = 0
                         # Try to add 2 edges
                         added = 0
                         for k in range(n):
                             for l in range(k+1, n):
                                 if not adj[k][l] and try_add_edge(k, l):
                                     adj[k][l] = 1
                                     adj[l][k] = 1
                                     added += 1
                                     if added == 2:
                                         break
                             if added == 2:
                                 break
                         if added < 2:
                             # Restore
                             adj[i][j] = 1
                             adj[j][i] = 1
                             # Undo added edges?
                             # This is getting complicated. 
                             # Let's stick to simple add/remove/swap.
                             pass

    # Count edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

construct()
