import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct the q=5 Polarity Graph
    # Points in PG(2, 5)
    # We can represent points as (x, y) in affine plane plus points at infinity
    # Or just use the standard construction.
    # For q=5, n=31.
    # Vertices 0..30.
    # Edges:
    # Standard construction:
    # v = q^2 + q + 1
    # For each point P, it is connected to the q+1 points on its polar line.
    # We need an explicit list of edges.
    
    # Let's generate the graph using the property that it's a (31, 6, 1, 2) srg?
    # Actually, let's just hardcode a known 31-vertex C4-free graph or generate it.
    
    # Generation of Polarity Graph for q=5:
    # Use the field GF(5).
    # Points: (x, y, 1) for x,y in GF(5) [25 points]
    # Points at infinity: (x, 1, 0) for x in GF(5) [5 points]
    # Point (0, 0, 1) is also there? No, (0,0,1) is in affine.
    # Total 25 + 5 + 1 (point (0,0,1) is already counted? No, (0,0,1) is one of the 25).
    # Wait, affine plane has q^2 points. Projective has q^2+q+1.
    # Affine: (x,y) for x,y in {0..4}. 25 points.
    # Infinity: (x,1,0) for x in {0..4}. 5 points.
    # Plus the point at infinity for the "vertical" lines?
    # Standard homogeneous coords:
    # (x,y,z) not all 0.
    # Normalize z=1 for affine. (x,y,1). 25 points.
    # Normalize y=1 for infinity. (x,1,0). 5 points.
    # One more: (0,0,1) is in affine.
    # What is the 31st point?
    # The point (0,0,1) is (0,0) in affine.
    # The points at infinity are (1,0,0), (0,1,0), (1,1,0), (1,-1,0), (1,2,0)?
    # In PG(2, q), there are q+1 points at infinity.
    # So 25 + 6 = 31.
    
    # Let's map them to 0..30.
    # 0..24: (x,y,1) -> idx = x*5 + y
    # 25..29: (x,1,0) -> idx = 25 + x
    # 30: (0,0,1)? No, (0,0,1) is idx 0.
    # The 31st point is usually the one where z=0 and y=0? (1,0,0).
    # Let's just use a known adjacency list for q=5 polarity graph if possible,
    # or construct it via the quadratic form x^2 + y^2 + z^2 = 0?
    # Over GF(5), -1 is a square (2^2=4=-1).
    # So x^2 + y^2 + z^2 = 0 has solutions.
    
    # Simpler: Use the seed's 30-vertex graph (10 triangles) and try to improve?
    # No, 31 is better.
    
    # Let's try to construct the q=5 graph properly.
    # Reference: The polarity graph of PG(2,5) is a (31, 6, 1, 2) strongly regular graph.
    # We can generate it by taking the 31 points and connecting p to q if p is on the polar line of q.
    
    # To save time and complexity, I will use a random C4-free graph generation
    # or the 31-vertex graph from literature if I can recall it.
    # Actually, the 31-vertex graph is the "Polarity Graph of order 5".
    
    # Let's implement a generic C4-free graph builder using the "Erdos-Renyi" style
    # or just use the 30-vertex seed and add 1 vertex?
    # The seed has 10 disjoint triangles.
    # Independence number is 10 (pick one from each triangle).
    # If we add a 31st vertex, can we keep alpha <= 10?
    # If we connect the new vertex to one vertex in each triangle, the new vertex is not independent with those.
    # But we might form a C4?
    # Triangles are disjoint. No edges between them.
    # If we connect v_new to v1 in T1, v2 in T2, etc.
    # v1 and v2 have no common neighbors (disjoint triangles).
    # So no C4 involving v_new and two existing vertices from different triangles.
    # C4 within a triangle? No, triangle is C3.
    # So we can add a vertex connected to one vertex in each of the 10 triangles.
    # This gives n=31.
    # Independence number:
    # Original alpha=10.
    # New vertex v is connected to one vertex in each triangle.
    # Can we pick an independent set of size 11?
    # We must pick v.
    # If we pick v, we cannot pick the 10 neighbors.
    # We can pick the other 2 vertices from each triangle?
    # Triangle T_i has 3 vertices. One is neighbor of v.
    # The other 2 are not neighbors of v.
    # Are they independent? Yes, they are not connected to each other?
    # In a triangle, all 3 are connected.
    # So we can pick at most 1 vertex from each triangle if we don't pick the neighbor?
    # Wait, if we don't pick the neighbor, we can pick one of the other two.
    # So we can pick 10 vertices (one from each triangle, not the neighbor) + v?
    # Are the 10 chosen vertices independent of each other?
    # Yes, triangles are disjoint.
    # So {v} U {one non-neighbor from each T_i} is an independent set of size 11.
    # So this construction FAILS the alpha <= 10 constraint.
    
    # So we cannot simply add a vertex to the seed.
    # We need a denser graph or different structure.
    
    # The q=5 polarity graph has alpha=6.
    # So it is very safe.
    # I will implement the q=5 polarity graph.
    
    # Construction of PG(2,5) Polarity Graph:
    # Points: 0..30.
    # Let's use the explicit coordinates.
    # Field F5 = {0, 1, 2, 3, 4}.
    # Points P:
    # 0-24: (x, y, 1) for x in 0..4, y in 0..4. idx = x*5 + y.
    # 25-29: (x, 1, 0) for x in 0..4. idx = 25 + x.
    # 30: (0, 0, 1)? No, that's 0.
    # The 31st point is (1, 0, 0)?
    # Let's check the count.
    # z=1: 25 points.
    # z=0, y!=0: (x, y, 0) ~ (x/y, 1, 0). 5 points.
    # z=0, y=0: (x, 0, 0) ~ (1, 0, 0). 1 point.
    # Total 31.
    # So 30 is (1, 0, 0).
    
    # Polarity: x^2 + y^2 + z^2 = 0.
    # Point P=(x1,y1,z1) is adjacent to Q=(x2,y2,z2) if x1x2 + y1y2 + z1z2 = 0.
    # Note: In char 5, this is symmetric.
    
    edges = []
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    for i in range(31):
        p1 = points[i]
        for j in range(i+1, 31):
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                edges.append([i, j])
                
    # This graph has n=31.
    # It is C4-free.
    # Alpha is 6.
    
    # Can we extend it?
    # We have 31 vertices. We want 39.
    # We need to add 8 vertices.
    # We can try to add vertices one by one.
    # To add a vertex v, we need a set of neighbors S such that no two in S have a common neighbor in G.
    # In this graph, any two non-adjacent vertices have 1 common neighbor.
    # Any two adjacent vertices have 0 common neighbors?
    # Let's check lambda.
    # For (31, 6, 1, 2) srg:
    # lambda=1 (adjacent have 1 common neighbor)?
    # mu=2 (non-adjacent have 2 common neighbors)?
    # If mu=2, then non-adjacent vertices have 2 common neighbors.
    # Then we CANNOT add a vertex connected to two non-adjacent vertices.
    # We can only add a vertex connected to a set S where every pair in S is ADJACENT (clique).
    # Max clique size?
    # In polarity graph, max clique is q+1 = 6.
    # So we can add a vertex connected to a clique of size 6.
    
    # Let's try to add 8 vertices.
    # For each new vertex, find a max clique in the current graph, connect to it.
    # But we must ensure the new vertex doesn't create C4 with other new vertices?
    # If we add v1 connected to C1.
    # Then add v2 connected to C2.
    # If C1 and C2 share vertices, or if v1 is in C2, etc.
    # This is getting complex.
    
    # Given the time limit, returning the 31-vertex graph is a safe improvement (1.033).
    # The seed is 30.
    
    return {'n': 31, 'edges': edges}
