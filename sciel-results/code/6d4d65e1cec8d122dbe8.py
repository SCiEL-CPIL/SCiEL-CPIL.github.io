import itertools
import numpy as np
import random
import time

def construct():
    # 1. Generate all points in PG(2, 5)
    points = []
    seen = set()
    inv_map = {1: 1, 2: 3, 3: 2, 4: 4}
    
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                vec = (x, y, z)
                idx = 0
                if vec[0] == 0:
                    idx = 1
                if vec[0] == 0 and vec[1] == 0:
                    idx = 2
                val = vec[idx]
                inv_val = inv_map[val]
                c = inv_val
                cx = (x * c) % 5
                cy = (y * c) % 5
                cz = (z * c) % 5
                canon = (cx, cy, cz)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
                    
    n_vertices = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # 2. Construct the polarity graph
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
                
    # Helper to count C4s in the current graph
    def count_c4s():
        count = 0
        # A C4 is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
        # We can count pairs of common neighbors for each pair of vertices
        # Number of C4s = sum over all pairs (u,v) of (common_neighbors(u,v) choose 2)
        # Wait, this counts each C4 twice?
        # A C4 u-a-v-b-u.
        # Pairs (u,v) and (a,b).
        # common_neighbors(u,v) includes a and b.
        # So for pair (u,v), we count 1.
        # For pair (a,b), we count 1.
        # Total 2.
        # So sum / 2 gives the number of C4s.
        
        # Optimization: Use bitsets or simple loops
        # n=31 is small.
        c4_count = 0
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                # Find common neighbors
                common = []
                for w in range(n_vertices):
                    if adj[u][w] and adj[v][w]:
                        common.append(w)
                k = len(common)
                if k >= 2:
                    c4_count += k * (k - 1) // 2
        return c4_count // 2

    # Initial C4 count
    current_c4s = count_c4s()
    
    # 3. Add the best edge
    # We need exactly 91 edges.
    # If we have 90, add 1.
    # Find non-edge with min C4s added.
    
    def count_c4s_added(u, v):
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
                    
    if best_edge:
        u, v = best_edge
        adj[u][v] = True
        adj[v][u] = True
        edges.append(best_edge)
        current_c4s = count_c4s()

    # 4. Local Search
    # Try to swap edges to reduce C4s
    # Time limit 120s.
    start_time = time.time()
    
    # Precompute edge list for faster iteration
    edge_set = set(map(tuple, edges))
    
    while time.time() - start_time < 110:
        improved = False
        # Shuffle edges to avoid local minima
        random.shuffle(edges)
        
        for e_idx, (u, v) in enumerate(edges):
            if time.time() - start_time > 110:
                break
            
            # Remove edge (u, v)
            adj[u][v] = False
            adj[v][u] = False
            edges.pop(e_idx)
            edge_set.discard((u, v))
            
            # Try to add a non-edge
            # Only try a subset of non-edges to save time
            # Or all non-edges? 31*30/2 = 465.
            # 465 * count_c4s (which is O(N^3) ~ 30k) = 14M ops per swap attempt.
            # Too slow.
            
            # We need a faster C4 count or heuristic.
            # Heuristic: Count C4s added by new edge.
            # But we also removed an edge, which might have been in C4s.
            # This is complex.
            
            # Alternative: Just try a few random non-edges.
            # Pick 20 random non-edges.
            non_edges = []
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    if not adj[i][j] and (i, j) != (u, v) and (j, i) != (u, v):
                        non_edges.append((i, j))
                        
            random.shuffle(non_edges)
            candidates = non_edges[:20]
            
            best_candidate = None
            best_score = current_c4s + 1 # Must be better than current
            
            # We need to evaluate C4s for each candidate.
            # This is slow.
            # Let's just accept if the added edge creates fewer C4s than the removed edge was in?
            # Hard to know.
            
            # Let's skip complex local search and just rely on the initial construction.
            # If the initial construction is correct, it should be good.
            
            # Restore edge
            edges.insert(e_idx, [u, v])
            edge_set.add((u, v))
            adj[u][v] = True
            adj[v][u] = True
            
        # If no improvement, break?
        # Actually, the local search above is flawed because it doesn't evaluate properly.
        # I will remove the local search to avoid bugs and rely on the polarity graph + best edge.
        break

    return edges
