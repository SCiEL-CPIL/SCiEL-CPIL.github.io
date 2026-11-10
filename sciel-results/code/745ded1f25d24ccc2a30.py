import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 10  # Safety margin, actual run is < 1s
    
    q = 5
    # Build PG(2, q) points
    # Points: (1, x, y) for x,y in GF(q); (0, 1, z) for z in GF(q); (0, 0, 1)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    # The polarity graph has 90 edges for q=5
    if len(edges) != 90:
        # Fallback if construction fails (should not happen)
        # Generate a random graph with 91 edges? No, better to fail gracefully or use a known good graph
        # But for this specific q, it is deterministic.
        pass

    # Find the best non-edge to add
    best_edge = None
    best_cost = float('inf')
    
    # Precompute adjacency lists as lists for faster iteration if needed, but sets are fine for n=31
    # We want to minimize the number of new 4-cycles.
    # New 4-cycles involving the new edge (u, v):
    # 1. u - a - v - b - u where a, b are common neighbors of u and v.
    #    Number of such cycles is C(k, 2) where k = |N(u) \cap N(v)|.
    # 2. u - a - b - v - u where a in N(u), b in N(v), a ~ b, a != b, a != v, b != u.
    #    Since u, v are non-adjacent, v not in N(u) and u not in N(v).
    #    So a in N(u), b in N(v), a ~ b, a != b.
    
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                continue
            
            common = adj[i] & adj[j]
            k = len(common)
            
            # Type 1: C(k, 2)
            cost1 = k * (k - 1) // 2
            
            # Type 2:
            cost2 = 0
            # Iterate over neighbors of i
            for a in adj[i]:
                # a cannot be j (j not in adj[i])
                # Check neighbors of a
                for b in adj[j]:
                    # b cannot be i
                    if b == a:
                        continue
                    if b in adj[a]:
                        cost2 += 1
            
            total_cost = cost1 + cost2
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_edge = [i, j]
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_edge is not None:
        edges.append(best_edge)
    
    # Ensure exactly 91 edges
    if len(edges) < 91:
        existing = set(tuple(sorted(e)) for e in edges)
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i, j) not in existing:
                    edges.append([i, j])
                    existing.add((i, j))
            if len(edges) >= 91:
                break
                
    return edges[:91]
