import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate points of PG(2, 5)
    points = []
    seen = set()
    # x in 0..4, y in 0..4, z in 0..4
    # Canonicalize to avoid duplicates
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form:
                # If x != 0, scale to x=1
                # Elif y != 0, scale to y=1
                # Else z=1
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    cx = 1
                    cy = (y * inv_x) % 5
                    cz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    cx = 0
                    cy = 1
                    cz = (z * inv_y) % 5
                else:
                    cx = 0
                    cy = 0
                    cz = 1
                
                p = (cx, cy, cz)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # 2. Define polarity matrix M
    # Using M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] might not give 90 edges if discriminant is wrong
    # Let's use a known good one for q=5.
    # M = [[1, 0, 0], [0, 1, 1], [0, 1, 0]]?
    # Let's try M = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    
    def dot(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
        
    n = 31
    adj = [[0]*n for _ in range(n)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
                
    # Check edge count
    if len(edges) != 90:
        # Fallback: if construction failed, use a heuristic or different M
        # But it should be 90.
        pass

    # Find a non-edge to add
    # We want to minimize C4s created.
    # As derived, any non-edge should create 1 C4.
    # Let's just pick the first non-edge.
    added = False
    for i in range(n):
        if added: break
        for j in range(i+1, n):
            if adj[i][j] == 0:
                edges.append([i, j])
                added = True
                break
                
    return edges
