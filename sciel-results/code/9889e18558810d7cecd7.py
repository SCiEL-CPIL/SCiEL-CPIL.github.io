import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # We will use a greedy approach with random restarts.
    # In each restart, we try to add edges one by one.
    # We maintain the adjacency matrix and check for K3,3 violations.
    
    # To speed up the K3,3 check, we can precompute or maintain some info,
    # but for n=20, a direct check is fast enough if optimized.
    
    def is_k33_free(adj):
        # Check if any 3 vertices have 3 common neighbors
        # This is equivalent to checking if the graph contains K3,3
        # A faster way: for every pair of vertices (u, v), count common neighbors.
        # If any pair has >= 3 common neighbors, it doesn't immediately mean K3,3,
        # but if we pick any 3 from those common neighbors, we have K3,3?
        # No. K3,3 is a bipartite subgraph with parts of size 3.
        # The condition "no three vertices have three common neighbours" is exactly
        # the condition that the graph is K3,3-free?
        # Let's re-read the task: "no three vertices have three common neighbours (no K3,3)"
        # Actually, if there are 3 vertices A,B,C and 3 vertices D,E,F such that
        # D,E,F are all neighbors of A,B,C, then {A,B,C} x {D,E,F} is a K3,3.
        # So the condition is: for any set S of 3 vertices, |N(S)| < 3 is false?
        # No, it says "no three vertices have three common neighbours".
        # This means for any distinct u,v,w, |N(u) ∩ N(v) ∩ N(w)| <= 2.
        # This is indeed equivalent to being K3,3-free.
        
        # Efficient check:
        # For each pair (u, v), let C = N(u) ∩ N(v).
        # If |C| >= 3, does that imply K3,3?
        # No. We need 3 vertices u,v,w such that their common neighborhood has size >= 3.
        # So we need to check triples.
        # Number of triples is C(20,3) = 1140.
        # For each triple, compute intersection of 3 sets of size up to 20.
        # This is very fast.
        
        for i in range(n):
            for j in range(i+1, n):
                # If we already know u,v,w, we can optimize.
                # But let's just iterate triples.
                pass
        
        # Better: iterate triples (i, j, k)
        for i in range(n):
            ni = adj[i]
            for j in range(i+1, n):
                nij = ni & adj[j]
                if nij.count() < 3:
                    continue
                for k in range(j+1, n):
                    if (nij & adj[k]).count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].count() for i in range(n)) // 2

    # Greedy construction
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        adj = [0] * n
        # Start with empty graph or random sparse graph?
        # Empty is safer for validity, but might get stuck in local minima.
        # Let's start empty.
        
        # Order of adding edges: random permutation of all possible edges
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            
            # Try adding edge (u, v)
            # Check if adding it creates a K3,3
            # We only need to check triples that include u or v?
            # No, adding an edge (u,v) only affects common neighborhoods of triples
            # that contain u or v?
            # Actually, common neighborhood of (a,b,c) changes only if u or v is in {a,b,c}
            # OR if u or v is a common neighbor of (a,b,c).
            # Wait, N(a) ∩ N(b) ∩ N(c).
            # Adding edge (u,v) adds v to N(u) and u to N(v).
            # So it can only increase the common neighborhood of triples involving u or v.
            # Specifically, if a triple (a,b,c) does not contain u or v, its common
            # neighborhood is unchanged.
            # If it contains u but not v: N(u) gains v. So the intersection gains v
            # only if v is in N(a) ∩ N(b) ∩ N(c) (where c is the other vertex).
            # So we only need to check triples containing u or v.
            
            # Triples containing u: (u, i, j) for i < j, i,j != u
            # Triples containing v: (v, i, j) for i < j, i,j != v
            # Total ~ 2 * C(19,2) = 2 * 171 = 342 checks.
            
            valid = True
            # Check triples with u
            for i in range(n):
                if i == u: continue
                for j in range(i+1, n):
                    if j == u: continue
                    # Triple (u, i, j)
                    # Common neighbors before: N(u) ∩ N(i) ∩ N(j)
                    # After adding (u,v): N(u) ∪ {v} ... wait, N(u) becomes N(u) ∪ {v}
                    # So new common = (N(u) ∩ N(i) ∩ N(j)) ∪ ({v} if v in N(i)∩N(j))
                    # We need new common size < 3.
                    # Old size = (adj[u] & adj[i] & adj[j]).count()
                    # If old size >= 3, it was already invalid (shouldn't happen if we maintain validity)
                    # If old size == 2, we must ensure v is NOT in N(i) ∩ N(j).
                    # If v is in N(i) ∩ N(j), then new size becomes 3 -> Invalid.
                    
                    common = adj[u] & adj[i] & adj[j]
                    if common.count() >= 3:
                        valid = False
                        break
                    if common.count() == 2:
                        # Check if v is common neighbor of i and j
                        if (adj[i] & adj[j]) & (1 << v):
                            valid = False
                            break
                if not valid: break
            if not valid: continue
            
            # Check triples with v
            for i in range(n):
                if i == v: continue
                for j in range(i+1, n):
                    if j == v: continue
                    common = adj[v] & adj[i] & adj[j]
                    if common.count() >= 3:
                        valid = False
                        break
                    if common.count() == 2:
                        if (adj[i] & adj[j]) & (1 << u):
                            valid = False
                            break
                if not valid: break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # End of one restart
        cur_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    cur_edges.append([i, j])
        
        if len(cur_edges) > best_score:
            best_score = len(cur_edges)
            best_edges = cur_edges
            
    return best_edges
