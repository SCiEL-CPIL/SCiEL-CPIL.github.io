import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Start with 8 disjoint K5s: 8 * 10 = 80 edges
    # Vertices 0-4: K5_0, 5-9: K5_1, ..., 35-39: K5_7
    adj = [[0] * n for _ in range(n)]
    edges = []
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
    
    # Helper: Check if adding edge (u,v) creates a C6
    # A new C6 must use the edge (u,v). So we look for a path u-x-y-z-v of length 4.
    # u, x, y, z, v must be distinct.
    # x in N(u), z in N(v), y in N(x) ∩ N(z).
    # y != u, y != v, y != x, y != z.
    # Also x != z, x != v, z != u (guaranteed if u,v not adjacent yet and graph simple).
    # Also x != y and z != y (guaranteed by y in N(x) and y in N(z) in simple graph? 
    # No, N(x) doesn't contain x. But y could be x? No, y is a neighbor of x, so y != x.
    # Similarly y != z.
    # So we just need to check distinctness of u, x, y, z, v.
    # u and v are fixed. x != u (x in N(u)), v != x (x != v).
    # z != v (z in N(v)), z != u (z != u).
    # x != z.
    # y != u, y != v, y != x, y != z.
    
    def would_create_c6(u, v):
        # Iterate over x in N(u)
        for x in range(n):
            if adj[u][x] == 0 or x == v:
                continue
            # Iterate over z in N(v)
            for z in range(n):
                if adj[v][z] == 0 or z == u:
                    continue
                if z == x:
                    continue
                # Check if x and z have a common neighbor y
                # y != u, y != v, y != x, y != z
                for y in range(n):
                    if adj[x][y] and adj[z][y]:
                        if y == u or y == v or y == x or y == z:
                            continue
                        return True
        return False

    best_edges = edges[:]
    best_count = len(best_edges)
    
    # Pre-compute list of all non-edges
    present = set()
    for e in edges:
        present.add((e[0], e[1]))
        present.add((e[1], e[0]))
    
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in present:
                non_edges.append((u, v))
    
    random.seed(42)
    
    # Greedy edge addition with random restarts
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Reset to base graph
        adj = [[0] * n for _ in range(n)]
        edges = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u][v] = 1
                    adj[v][u] = 1
                    edges.append([u, v])
        
        # Shuffle non-edges
        candidates = non_edges[:]
        random.shuffle(candidates)
        
        # Try to add edges greedily
        added_any = True
        while added_any and time.time() < deadline - 15:
            added_any = False
            i = 0
            while i < len(candidates) and time.time() < deadline - 15:
                u, v = candidates[i]
                if adj[u][v] == 0:  # Not already added
                    if not would_create_c6(u, v):
                        adj[u][v] = 1
                        adj[v][u] = 1
                        edges.append([u, v])
                        added_any = True
                        # Remove from candidates to avoid re-checking
                        candidates.pop(i)
                        continue
                i += 1
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            if best_count >= 105:
                break
    
    return best_edges
