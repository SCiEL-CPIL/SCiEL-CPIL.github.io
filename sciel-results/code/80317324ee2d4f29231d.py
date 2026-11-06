import itertools
import numpy as np

def construct():
    # 1. Generate all points in PG(2, 5)
    # Points are equivalence classes of non-zero vectors in GF(5)^3
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonicalize: make first non-zero element 1
                # Inverse in GF(5): 1->1, 2->3, 3->2, 4->4
                inv_map = {1: 1, 2: 3, 3: 2, 4: 4}
                
                vec = (x, y, z)
                # Find first non-zero
                idx = 0
                if vec[0] == 0:
                    idx = 1
                if vec[0] == 0 and vec[1] == 0:
                    idx = 2
                
                val = vec[idx]
                inv_val = inv_map[val]
                
                # Scale vector by inv_val
                c = inv_val
                cx = (x * c) % 5
                cy = (y * c) % 5
                cz = (z * c) % 5
                
                canon = (cx, cy, cz)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    n_vertices = len(points)
    
    # 2. Construct the polarity graph (90 edges)
    # Edge if dot product is 0
    adj = [[False] * n_vertices for _ in range(n_vertices)]
    edges = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
                
    # Current edge count should be 90
    # 3. Find the non-edge that adds the fewest C4s
    # Number of C4s added by (u,v) is number of edges between N(u) and N(v)
    
    def count_c4s_added(u, v):
        # u, v are not adjacent
        count = 0
        for a in range(n_vertices):
            if adj[u][a]:
                for b in range(n_vertices):
                    if adj[v][b] and a < b: # avoid double counting if we iterated all pairs? 
                        # Actually, we just need to count edges (a,b) where a in N(u), b in N(v)
                        # It's safer to just check if adj[a][b] is true
                        # But a and b are specific vertices.
                        # We iterate all a in N(u) and all b in N(v).
                        # If a==b, we can't have a self loop, so adj[a][b] is False.
                        pass
        
        # Better implementation:
        neighbors_u = [a for a in range(n_vertices) if adj[u][a]]
        neighbors_v = [b for b in range(n_vertices) if adj[v][b]]
        
        count = 0
        for a in neighbors_u:
            for b in neighbors_v:
                if a != b and adj[a][b]:
                    count += 1
        return count

    min_c4s = float('inf')
    best_edge = None
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if not adj[i][j]:
                c4_count = count_c4s_added(i, j)
                if c4_count < min_c4s:
                    min_c4s = c4_count
                    best_edge = [i, j]
                    
    # Add the best edge
    if best_edge is not None:
        edges.append(best_edge)
        
    return edges
