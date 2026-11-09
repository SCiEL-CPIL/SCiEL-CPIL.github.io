import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    def is_c4_free_after_add(adj, u, v):
        # Check if adding (u, v) creates a C4
        # C4 is u-x-y-v-u
        # We need to check if there is a path of length 3 from u to v
        # i.e., exists x in N(u) and y in N(v) such that (x, y) is an edge
        # Note: x != y and x != v and y != u are guaranteed if graph is simple and we check edge existence
        for x in adj[u]:
            for y in adj[v]:
                if x == y:
                    continue
                if y in adj[x]:
                    return False
        return True

    def min_degree(adj, n):
        return min(len(adj[i]) for i in range(n))

    def try_construct(n, target_min_deg, max_restarts=500):
        # Try to build a C4-free graph on n vertices with min degree >= target_min_deg
        # Using greedy with randomization
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            adj = [set() for _ in range(n)]
            # We will try to add edges
            # To help satisfy min degree, we can start with a random matching or just random edges
            # But greedy from empty is standard.
            
            # Generate all possible edges
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            added_edges = []
            
            # Greedy addition
            for u, v in edges:
                if time.time() > deadline:
                    return None
                if u in adj[v]:
                    continue
                if is_c4_free_after_add(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added_edges.append((u, v))
            
            # Check min degree
            md = min_degree(adj, n)
            if md >= target_min_deg:
                return added_edges
        
        return None

    # Target n=46, min degree 7
    # n=46 - 39 = 7
    edges_46 = try_construct(46, 7, max_restarts=100)
    if edges_46 is not None:
        return {'n': 46, 'edges': [list(e) for e in edges_46]}
    
    # Target n=45, min degree 6
    # n=45 - 39 = 6
    edges_45 = try_construct(45, 6, max_restarts=100)
    if edges_45 is not None:
        return {'n': 45, 'edges': [list(e) for e in edges_45]}
        
    # Target n=44, min degree 5
    edges_44 = try_construct(44, 5, max_restarts=100)
    if edges_44 is not None:
        return {'n': 44, 'edges': [list(e) for e in edges_44]}

    # Fallback to n=39
    return {'n': 39, 'edges': []}
