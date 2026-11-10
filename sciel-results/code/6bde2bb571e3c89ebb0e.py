import time

def construct():
    start_time = time.time()
    
    q = 5
    points = []
    # Points in PG(2, q)
    # (1, x, y) for x, y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # (0, 1, z) for z in GF(q)
    for z in range(q):
        points.append((0, 1, z))
    # (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points) # 31
    
    # Precompute dot products or just compute on fly
    # dot(p, r) = (p0*r0 + p1*r1 + p2*r2) % q
    
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            r = points[j]
            d = (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
            if d == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback: if construction failed, return random 91 edges?
        # But it should be 90.
        pass

    best_cost = float('inf')
    best_edge = None
    
    # Find non-edge with minimum new 4-cycles
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                continue
            
            # Calculate cost: number of edges between N(i) and N(j)
            cost = 0
            ni = adj[i]
            nj = adj[j]
            
            for x in ni:
                # Check if x is connected to any y in nj
                # x in adj[y]
                for y in nj:
                    if x in adj[y]:
                        cost += 1
                        
            if cost < best_cost:
                best_cost = cost
                best_edge = [i, j]
                
            # Optional: break early if cost is 0? 
            # If cost is 0, we can't get better.
            if best_cost == 0:
                break
        if best_cost == 0:
            break
            
    if best_edge is not None:
        edges.append(best_edge)
        
    # Ensure exactly 91 edges
    if len(edges) < 91:
        # Add arbitrary non-edges if needed
        existing = set(map(tuple, edges))
        for i in range(n):
            for j in range(i + 1, n):
                if len(edges) >= 91:
                    break
                if (i, j) not in existing:
                    edges.append([i, j])
                    existing.add((i, j))
            if len(edges) >= 91:
                break
                
    return edges[:91]
