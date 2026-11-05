import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 31
    m = 91
    
    # Generate points of PG(2, 5)
    # Points are 1D subspaces of F_5^3
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: first non-zero is 1
                if a != 0:
                    inv_a = pow(a, -1, 5)
                    p = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    inv_b = pow(b, -1, 5)
                    p = (0, 1, (c * inv_b) % 5)
                else:
                    inv_c = pow(c, -1, 5)
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # Map points to indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Construct polarity graph with A = Identity Matrix
    # Adjacency: P . Q = 0 mod 5
    edges = []
    adj = [[False]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            p_i = points[i]
            p_j = points[j]
            dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i][j] = True
                adj[j][i] = True
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback if something went wrong with the specific polarity
        # Just use lexicographic for safety, though it should be 90
        # But we expect 90.
        pass
        
    # Count C4s to verify 0
    # A C4 is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # Or simply, number of pairs of vertices with >= 2 common neighbors?
    # No, C4 count is sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 2 ?
    # Actually, each C4 has 2 pairs of opposite vertices.
    # So C4_count = sum_{u<v} C(common_neighbors(u,v), 2) / 2
    # Wait, if u and v are not adjacent, they are opposite in the C4?
    # In a C4 a-b-c-d-a, a and c are opposite, b and d are opposite.
    # So yes, for each pair of non-adjacent vertices (which are opposite in the C4),
    # the number of common neighbors is the number of paths of length 2.
    # Each C4 contributes 1 to the count for pair (a,c) and 1 for pair (b,d).
    # So Total C4s = 1/2 * sum_{u,v} C(common(u,v), 2).
    
    c4_count = 0
    common_neighbors = {}
    for u in range(n):
        for v in range(u+1, n):
            cn = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    cn += 1
            common_neighbors[(u,v)] = cn
            if cn >= 2:
                c4_count += cn * (cn - 1) // 2
                
    c4_count //= 2
    
    if c4_count != 0:
        # If not C4-free, we have a problem. 
        # But polarity graphs are C4-free.
        pass

    # Find the non-edge with minimum common neighbors
    min_cn = float('inf')
    best_edge = None
    
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]:
                cn = common_neighbors[(u,v)]
                if cn < min_cn:
                    min_cn = cn
                    best_edge = [u, v]
                    
    if best_edge is None:
        # Should not happen
        return edges[:91]
        
    edges.append(best_edge)
    
    return edges
