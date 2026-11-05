import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 40
    
    # Start with 8 disjoint K5s
    adj = [[False] * n for _ in range(n)]
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    
    # Helper to check if adding edge (u,v) creates a C6
    def would_create_c6(u, v):
        # C6 = u - a - b - c - d - v
        # Need to check if there's a path of length 5 between u and v
        # BFS from u to depth 2, BFS from v to depth 3, check intersection
        # Or more efficiently: for each neighbor a of u, for each neighbor b of a (b != u),
        # for each neighbor c of b (c != a, b), check if c is connected to d which is connected to v
        
        # More efficient: precompute neighbors
        # Path: u - a - b - c - d - v
        # a in N(u), b in N(a)\{u}, c in N(b)\{a}, d in N(c)\{b} ∩ N(v)
        
        # Let's use a different approach:
        # For the path u-a-b-c-d-v to exist:
        # a is neighbor of u
        # b is neighbor of a (b != u)
        # c is neighbor of b (c != a)
        # d is neighbor of c (d != b) AND d is neighbor of v
        
        # We can iterate over a in N(u), then b in N(a)\{u}, then c in N(b)\{a},
        # and check if c has a neighbor d that is also a neighbor of v (d != b)
        
        for a in range(n):
            if not adj[u][a]:
                continue
            for b in range(n):
                if b == u or not adj[a][b]:
                    continue
                for c in range(n):
                    if c == a or not adj[b][c]:
                        continue
                    # Check if c is connected to some d that is connected to v (d != b)
                    # d must be neighbor of c and neighbor of v, d != b
                    found = False
                    for d in range(n):
                        if d == b:
                            continue
                        if adj[c][d] and adj[v][d]:
                            found = True
                            break
                    if found:
                        return True
        return False
    
    # Local search: try to add random edges, remove edges to resolve conflicts
    current_edges = set(map(tuple, edges))
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edge_count = len(edges)
    best_edges = edges[:]
    
    while time.time() < deadline:
        # Try to add a random non-edge
        u, v = random.choice(all_edges)
        if (u, v) in current_edges or (v, u) in current_edges:
            continue
        
        if not would_create_c6(u, v):
            # Add the edge
            adj[u][v] = adj[v][u] = True
            current_edges.add((u, v))
            edges.append([u, v])
            
            if len(edges) > best_edge_count:
                best_edge_count = len(edges)
                best_edges = edges[:]
        else:
            # Try to find a swap: remove one edge, add (u,v)
            # Try removing a random edge from the current graph
            if edges:
                remove_idx = random.randint(0, len(edges) - 1)
                ru, rv = edges[remove_idx]
                
                # Temporarily remove
                adj[ru][rv] = adj[rv][ru] = False
                current_edges.discard((ru, rv))
                current_edges.discard((rv, ru))
                
                if not would_create_c6(u, v):
                    # Add (u,v)
                    adj[u][v] = adj[v][u] = True
                    current_edges.add((u, v))
                    edges[remove_idx] = [u, v]
                else:
                    # Put back
                    adj[ru][rv] = adj[rv][ru] = True
                    current_edges.add((ru, rv))

    return best_edges
