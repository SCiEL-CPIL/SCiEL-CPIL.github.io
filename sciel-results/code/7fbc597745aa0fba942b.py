import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    q = 5
    n_base = q*q + q + 1
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
        
    def bilinear(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*y2 + y1*z2 + z1*x2) % q

    edges = []
    
    # Base edges
    for i in range(n_base):
        if time.time() > deadline:
            break
        for j in range(i + 1, n_base):
            if time.time() > deadline:
                break
            if bilinear(points[i], points[j]) != 0:
                edges.append([i, j])
                
    # Find 12 distinct triangles in Base
    # A triangle is 3 mutually adjacent vertices
    triangles = []
    for i in range(n_base):
        if time.time() > deadline:
            break
        for j in range(i + 1, n_base):
            if time.time() > deadline:
                break
            # Check if i, j are adjacent
            if bilinear(points[i], points[j]) == 0:
                continue
            for k in range(j + 1, n_base):
                if time.time() > deadline:
                    break
                # Check if i, k and j, k are adjacent
                if bilinear(points[i], points[k]) != 0 and bilinear(points[j], points[k]) != 0:
                    triangles.append((i, j, k))
                    if len(triangles) >= 12:
                        break
            if len(triangles) >= 12:
                break
        if len(triangles) >= 12:
            break
            
    # If we didn't find 12, we might have fewer, but PG(2,5) has many.
    # If less than 12, we can reuse? No, must be distinct to avoid shared neighbors?
    # If we reuse a triangle, two new vertices connected to same triangle share 3 neighbors.
    # If they are adjacent, they share 3 common neighbors -> C4.
    # So they must be non-adjacent if they share neighbors.
    # In our New subgraph (4 disjoint triangles), vertices in different triangles are non-adjacent.
    # So we CAN reuse triangles for vertices in different new-triangles.
    # But to be safe, let's just find as many as possible.
    
    n_new = 12
    n_total = n_base + n_new
    
    # New subgraph: 4 disjoint triangles
    # Indices: 31,32,33 | 34,35,36 | 37,38,39 | 40,41,42
    new_triangles = [
        [31, 32, 33],
        [34, 35, 36],
        [37, 38, 39],
        [40, 41, 42]
    ]
    
    for tri in new_triangles:
        if time.time() > deadline:
            break
        for a in range(3):
            for b in range(a + 1, 3):
                edges.append([tri[a], tri[b]])
                
    # Connect new vertices to base triangles
    # We have 12 new vertices.
    # We have `len(triangles)` base triangles.
    # If we have at least 12, assign distinct ones.
    # If not, assign cyclically, but ensure adjacent new vertices don't share base neighbors.
    # Since new triangles are disjoint, vertices in different new triangles are not adjacent.
    # So they can share base neighbors.
    # Vertices in the SAME new triangle are adjacent.
    # So they MUST have disjoint base neighbor sets.
    # So within each new triangle, the 3 vertices must be connected to 3 DISJOINT base triangles.
    # So we need 3 * 4 = 12 distinct base triangles.
    
    if len(triangles) < 12:
        # Fallback: reduce n? Or just use available?
        # If we don't have 12, we can't make 4 disjoint triangles with distinct connections.
        # We can make fewer new vertices?
        # Or make new subgraph different?
        # For now, assume we found 12.
        pass
        
    for i, v_new in enumerate(range(n_base, n_total)):
        if time.time() > deadline:
            break
        # Assign base triangle
        # We need to ensure that for any new triangle, its 3 vertices get 3 distinct base triangles.
        # Since we iterate v_new 31..42, and new_triangles are grouped by 3.
        # We can just assign triangles[i % len(triangles)]?
        # No, if i=0,1,2 are in same new triangle, they must get distinct base triangles.
        # So we should assign triangles[0], triangles[1], triangles[2] to 31,32,33.
        # triangles[3], triangles[4], triangles[5] to 34,35,36.
        
        tri_idx = i % len(triangles)
        base_tri = triangles[tri_idx]
        for v_base in base_tri:
            edges.append([v_new, v_base])
            
    return {'n': n_total, 'edges': edges}
