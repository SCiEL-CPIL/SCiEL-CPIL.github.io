import numpy as np

def construct():
    # GF(4) arithmetic
    # Elements: 0, 1, 2, 3
    # 2 is alpha, 3 is alpha^2
    # add is XOR
    # mul table
    mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(a, b):
        return a ^ b
        
    def gf_mul(a, b):
        return mul[a][b]
        
    def gf_dot(u, v):
        # u, v are tuples (x, y, z)
        # dot = x1*z2 + y1*y2 + z1*x2
        return gf_add(gf_mul(u[0], v[2]), gf_add(gf_mul(u[1], v[1]), gf_mul(u[2], v[0])))
        
    # Generate points of PG(2, 4)
    points = []
    # x = 1
    for y in range(4):
        for z in range(4):
            points.append((1, y, z))
    # x = 0, y = 1
    for z in range(4):
        points.append((0, 1, z))
    # x = 0, y = 0, z = 1
    points.append((0, 0, 1))
    
    # Map points to indices 0..20
    point_index = {p: i for i, p in enumerate(points)}
    
    # Build adjacency list
    # u ~ v if u != v and gf_dot(u, v) == 0
    adj = {i: set() for i in range(21)}
    for i in range(21):
        for j in range(i+1, 21):
            if gf_dot(points[i], points[j]) == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    # Check degree
    # Should be 4 for all vertices in PG(2,4) polarity graph
    # Let's verify one: (1,0,0)
    # dot((1,0,0), (x,y,z)) = 1*z + 0 + 0*x = z.
    # So neighbors have z=0.
    # Points with z=0: (1,0,0), (1,1,0), (1,2,0), (1,3,0), (0,1,0).
    # (1,0,0) is itself.
    # Neighbors: (1,1,0), (1,2,0), (1,3,0), (0,1,0).
    # Degree 4. Correct.
    
    # We need 20 vertices. Remove vertex 20 (which is (0,0,1)).
    # Neighbors of (0,0,1):
    # dot((0,0,1), (x,y,z)) = 0*z + 0*y + 1*x = x.
    # Neighbors have x=0.
    # Points with x=0: (0,1,0), (0,1,1), (0,1,2), (0,1,3), (0,0,1).
    # (0,0,1) is itself.
    # Neighbors: (0,1,0), (0,1,1), (0,1,2), (0,1,3).
    # Degree 4.
    
    # Create list of vertices to keep: 0..19
    kept_vertices = list(range(20))
    
    # Create edges
    edges = []
    edge_id = 0
    for i in kept_vertices:
        for j in adj[i]:
            if j in kept_vertices and i < j:
                # Assign color
                # We have 38 edges. Colors 0..19.
                # Just use edge_id % 20
                c = edge_id % 20
                edges.append([i, j, c])
                edge_id += 1
                
    return edges
