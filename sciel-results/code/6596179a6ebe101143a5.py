import itertools
import numpy as np

def construct():
    # Generate all 31 points of PG(2, 5)
    # Points are non-zero vectors (x, y, z) in GF(5)^3, normalized such that the first non-zero component is 1.
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first non-zero
                if x != 0:
                    # scale by x^-1
                    # x is 1..4
                    inv_x = pow(x, -1, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    n = 31
    
    # Polarity matrix M = diag(1, 1, 4) since -1 = 4 mod 5
    # Condition for adjacency: u^T M v = 0 mod 5
    # u = (u1, u2, u3), v = (v1, v2, v3)
    # u1*v1 + u2*v2 + 4*u3*v3 == 0
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            val = (u[0]*v[0] + u[1]*v[1] + 4*u[2]*v[2]) % 5
            if val == 0:
                edges.append([i, j])
                
    # We expect 90 edges
    # Build adjacency bitsets for fast common neighbor counting
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v) if u < v else (v, u))
        
    # Find the non-edge that adds the fewest C4s
    # Adding edge (u, v) adds C4s of the form u-a-v-b-u
    # This requires a, b to be common neighbors of u and v.
    # Number of new C4s = C(k, 2) where k is number of common neighbors.
    
    min_c4s = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set:
                continue
            # Count common neighbors
            common_mask = adj[i] & adj[j]
            k = bin(common_mask).count('1')
            new_c4s = k * (k - 1) // 2
            if new_c4s < min_c4s:
                min_c4s = new_c4s
                best_edge = [i, j]
                
    if best_edge:
        edges.append(best_edge)
        
    # Ensure we have exactly 91 edges
    # If the logic above failed for some reason (e.g. no non-edges, which is impossible for 31 verts 90 edges),
    # we fallback to adding arbitrary non-edges.
    if len(edges) < 91:
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    edges.append([i, j])
                    edge_set.add((i, j))
                    if len(edges) == 91:
                        break
            if len(edges) == 91:
                break
                
    return edges
