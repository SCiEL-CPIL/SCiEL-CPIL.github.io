import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
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
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    # Note: For q odd, some points are self-orthogonal. We exclude self-loops.
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                edges.add((i, j))
    
    # Target edges: 91
    target_edges = 91
    
    # Helper to count C4s involving a specific new edge (u, v)
    # This is the number of paths u-b-a-v where (v,a), (a,b), (b,u) are edges
    # And a != b, a != u, b != v
    def count_new_c4s(u, v):
        # Neighbors of u and v
        # We need to check if (a, b) is an edge
        # a is neighbor of v
        # b is neighbor of u
        # and a != b
        count = 0
        # Iterate over neighbors of v
        # To be fast, precompute neighbor sets
        # But we are in a loop, so we can just use the edges set
        # deg is small
        for a in range(n):
            if a == v or a == u: continue
            if (min(v, a), max(v, a)) not in edges: continue
            for b in range(n):
                if b == u or b == v: continue
                if b == a: continue
                if (min(u, b), max(u, b)) not in edges: continue
                if (min(a, b), max(a, b)) in edges:
                    count += 1
        return count

    # If we have more than target edges, we should remove some
    # But polarity graph usually has ~65 edges, so we likely need to add.
    
    while len(edges) < target_edges:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_score = float('inf')
        
        # Find all non-edges
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in edges:
                    continue
                # Calculate cost
                cost = count_new_c4s(i, j)
                if cost < best_score:
                    best_score = cost
                    best_edge = (i, j)
        
        if best_edge is None:
            break
            
        edges.add(best_edge)
        
        # If we run out of time, break
        if time.time() > deadline:
            break

    # If we have more than target edges (unlikely for polarity graph), remove greedily
    # But let's assume we are under.
    
    # Convert to list of lists
    result = [[u, v] for u, v in edges]
    return result
