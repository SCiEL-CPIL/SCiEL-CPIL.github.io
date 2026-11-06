import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Field F_5
    q = 5
    
    # Generate all points in PG(2, 5)
    points = []
    for x0 in range(q):
        for x1 in range(q):
            for x2 in range(q):
                if x0 == 0 and x1 == 0 and x2 == 0:
                    continue
                if x0 != 0:
                    inv_x0 = pow(x0, q-2, q)
                    p = (1, (x1 * inv_x0) % q, (x2 * inv_x0) % q)
                elif x1 != 0:
                    inv_x1 = pow(x1, q-2, q)
                    p = (0, 1, (x2 * inv_x1) % q)
                else:
                    inv_x2 = pow(x2, q-2, q)
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    n = len(points)
    if n != 31:
        points = [(i, 0, 0) for i in range(1, 5)] + [(0, i, 0) for i in range(1, 5)] + [(0, 0, i) for i in range(1, 5)]
        pass

    # Build adjacency matrix
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        u = points[i]
        for j in range(i+1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    if len(edges) > 90:
        while len(edges) > 90:
            edges.pop()
            adj = [[0]*n for _ in range(n)]
            for e in edges:
                adj[e[0]][e[1]] = 1
                adj[e[1]][e[0]] = 1
    elif len(edges) < 90:
        pass

    # Rebuild adj to be safe
    adj = [[0]*n for _ in range(n)]
    for e in edges:
        adj[e[0]][e[1]] = 1
        adj[e[1]][e[0]] = 1
    
    # Convert to numpy for fast computation
    A = np.array(adj, dtype=np.int64)
    
    # Find the non-edge that creates the fewest new 4-cycles
    # New 4-cycles involving edge (u,v) = number of paths of length 3 from u to v
    # = (A^3)[u][v] where A is adjacency matrix
    # But we need to be careful: A^3[u][v] counts walks of length 3, which includes
    # walks that may revisit vertices. For simple paths of length 3 (u-a-b-v with all distinct),
    # we need to subtract cases where vertices repeat.
    # 
    # Actually, for counting 4-cycles created by adding edge (u,v), we need to count
    # the number of distinct paths u-a-b-v where a,b are distinct from u,v and from each other.
    # This equals (A^3)[u][v] minus paths where vertices repeat.
    # 
    # A path u-a-b-v has vertices repeating if:
    # - a = v: then u-v-b-v, but a=v means adj[u][v]=1 which is false (non-edge)
    # - b = u: then u-a-u-v, adj[a][u]=1 and adj[u][v]=0, so no
    # - a = b: then u-a-a-v, requires adj[a][a]=1, but no self-loops
    # - a = v: impossible as above
    # - b = v: then u-a-v-v, requires adj[v][v]=1, no self-loops
    # 
    # So for non-edges (u,v), (A^3)[u][v] should equal the number of simple paths of length 3.
    
    A3 = A @ A @ A
    
    min_cost = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if adj[i][j] == 0:
                count = int(A3[i][j])
                if count < min_cost:
                    min_cost = count
                    best_edge = [i, j]
                    
    if best_edge is None:
        best_edge = [0, 1]
        
    edges.append(best_edge)
    
    return edges

# EVOLVE-BLOCK-END
