import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_c4_free(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & adj[j]: # If they share any neighbor, it's not C4? 
                    # No, C4 is u-v-w-x-u. 
                    # i, j are u, v. Common neighbors are w, x?
                    # If i and j are adjacent, common neighbors form triangles.
                    # C4 check: for any pair (u, v), |N(u) intersection N(v)| <= 1.
                    pass
        
        # Correct C4 check:
        # A C4 exists if there are two distinct vertices u, v that have at least 2 common neighbors.
        # This applies regardless of whether u, v are adjacent or not.
        for i in range(n):
            for j in range(i+1, n):
                if len(adj[i] & adj[j]) >= 2:
                    return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0

    def validate(n, edges):
        if not is_c4_free(n, edges):
            return False
        return min_degree(n, edges) >= n - 39

    # Try n=47 (likely fails, but fast check)
    # Try n=46 with robust search
    
    def find_graph(n, min_deg_target, max_time=10.0):
        best_edges = None
        while time.time() < start_time + max_time:
            # Randomized Greedy with Bitmask
            adj = [0] * n # Bitmasks
            edges = []
            
            # Order vertices by random priority?
            # Or just random edge order
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates C4
                # C4 is created if there exists w such that w~u and w~v.
                # We need to ensure that for any w, we don't have two common neighbors.
                # Actually, the condition is: |N(u) intersection N(v)| <= 1.
                # Currently, N(u) and N(v) are defined by adj[u] and adj[v].
                # If we add (u, v), we add v to N(u) and u to N(v).
                # This might create a C4 involving u, v and some existing w?
                # No, the C4 must be formed by the new edge.
                # A C4 containing (u, v) would be u-v-w-x-u.
                # This requires w ~ v and x ~ u, and w ~ x.
                # Or u-v-x-w-u?
                # The standard check: 
                # Adding (u, v) creates a C4 if there is a vertex w such that 
                # w is adjacent to u and v? No, that's a triangle.
                # w is adjacent to u, and x is adjacent to v, and w~x?
                
                # Efficient check:
                # For each w in N(u), if w is in N(v), then we have a common neighbor.
                # We allow 1 common neighbor. If we already have 1, adding (u, v) doesn't change common neighbors of (u, v)?
                # Wait. The condition is on pairs of vertices.
                # Adding (u, v) changes the adjacency of u and v.
                # It creates a C4 if:
                # 1. There exists w, x such that w~u, x~v, w~x, w!=v, x!=u.
                #    This is a C4 u-w-x-v-u.
                #    This requires w in N(u) and x in N(v) and w~x.
                
                # This is expensive to check for all w, x.
                # Alternative: 
                # Iterate through all w in N(u).
                # For each w, check if w is in N(v). If yes, count common neighbors.
                # If count >= 2, it's not a C4 caused by this edge?
                # No, the C4 is formed by the new edge (u, v) and existing edges.
                # The C4 is u-v-w-x-u.
                # Edges: (u, v) [new], (v, w) [existing], (w, x) [existing], (x, u) [existing].
                # So we need w ~ v and x ~ u and w ~ x.
                
                # Let's check if adding (u, v) creates a C4.
                # For each w in N(v):
                #   If w is in N(u):
                #      Then u, v, w form a triangle. Not a C4.
                #   Else:
                #      Check if there is x in N(u) such that x ~ w.
                #      If so, u-w-x-v-u is a C4.
                
                # This is O(deg(u) * deg(v)) roughly.
                
                ok = True
                # Optimization: if deg(u) or deg(v) is small, it's fast.
                for w in range(n):
                    if (adj[v] >> w) & 1: # w is neighbor of v
                        if (adj[u] >> w) & 1: # w is neighbor of u
                            continue # Triangle
                        # Check if w shares a neighbor with u
                        # x must be in N(u) and N(w)
                        # We can just check if N(u) & N(w) is non-empty.
                        # But we need to be careful.
                        # Actually, if we add (u, v), we just need to check if the new edge closes a cycle of length 4.
                        # The cycle is u-v-w-x-u.
                        # w is in N(v). x is in N(u). w~x.
                        # We can iterate w in N(v).
                        # For each w, check if N(u) & N(w) is not empty.
                        # If it is, C4 found.
                        
                        # Note: x cannot be v (v not in N(u) yet).
                        # w cannot be u (u not in N(v) yet).
                        
                        # So:
                        common = adj[u] & adj[w]
                        if common:
                            ok = False
                            break
                if ok:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
            
            degs = [bin(adj[i]).count('1') for i in range(n)]
            if min(degs) >= min_deg_target:
                return edges
        return None

    # Try n=46
    edges46 = find_graph(46, 7, max_time=50.0)
    if edges46 and validate(46, edges46):
        return {'n': 46, 'edges': edges46}
        
    # Try n=45
    edges45 = find_graph(45, 6, max_time=20.0)
    if edges45 and validate(45, edges45):
        return {'n': 45, 'edges': edges45}

    return {'n': 39, 'edges': []}
