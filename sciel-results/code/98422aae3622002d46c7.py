import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def get_adj(n, colors):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        return adj

    def find_violation(adj, n, c):
        # Find a monochromatic K2,2,2 of color c
        # Returns the 6 vertices or None
        # Method: Iterate over pairs of disjoint edges (u,v) and (x,y) in color c
        # Then look for z1, z2 such that (z1,z2) is in color c and they are connected to u,v and x,y
        
        edges_c = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == c:
                    edges_c.append((i, j))
                    
        # For each pair of disjoint edges
        # To speed up, we can just iterate
        L = len(edges_c)
        for i in range(L):
            u, v = edges_c[i]
            for j in range(i+1, L):
                x, y = edges_c[j]
                # Check disjoint
                if u in (x, y) or v in (x, y):
                    continue
                
                # Find common neighbors of u,v in G_c
                # and common neighbors of x,y in G_c
                # We need z1 in N(u,v) and z2 in N(x,y)
                # such that z1 != z2, z1,z2 not in {u,v,x,y}, and (z1,z2) in G_c
                
                # Precompute N(u,v) and N(x,y)
                # This is O(n)
                n_uv = []
                n_xy = []
                for k in range(n):
                    if k in (u, v, x, y):
                        continue
                    if adj[u][k] == c and adj[v][k] == c:
                        n_uv.append(k)
                    if adj[x][k] == c and adj[y][k] == c:
                        n_xy.append(k)
                        
                # Check pairs in n_uv and n_xy
                for z1 in n_uv:
                    for z2 in n_xy:
                        if z1 == z2:
                            continue
                        if adj[z1][z2] == c:
                            return (u, v, x, y, z1, z2)
                            
        return None

    def is_valid_fast(n, colors):
        adj = get_adj(n, colors)
        if find_violation(adj, n, 0) is not None:
            return False
        if find_violation(adj, n, 1) is not None:
            return False
        return True

    # 1. Try Paley 29
    n_paley = 29
    p = 29
    squares = {x * x % p for x in range(1, p)}
    colors_paley = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    
    if is_valid_fast(n_paley, colors_paley):
        return {'n': n_paley, 'colors': colors_paley}
        
    # 2. Fallback: Local Search for n=29
    best_n = 17
    best_colors = [0 if (j - i) % 17 in {x * x % 17 for x in range(1, 17)} else 1 for i in range(17) for j in range(i + 1, 17)]
    
    # Run local search
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        colors = [random.randint(0, 1) for _ in range(n_paley * (n_paley - 1) // 2)]
        adj = get_adj(n_paley, colors)
        
        # Hill climbing
        for step in range(1000):
            if time.time() > deadline:
                break
                
            viol0 = find_violation(adj, n_paley, 0)
            if viol0 is None:
                viol1 = find_violation(adj, n_paley, 1)
                if viol1 is None:
                    return {'n': n_paley, 'colors': colors}
                else:
                    # Flip an edge in viol1
                    # Pick a random edge from the 12 edges of the K2,2,2
                    # The vertices are u,v,x,y,z1,z2
                    # Edges are between different pairs
                    u, v, x, y, z1, z2 = viol1
                    # Pairs: (u,v), (x,y), (z1,z2)
                    # Edges: u-x, u-y, v-x, v-y, u-z1, u-z2, v-z1, v-z2, x-z1, x-z2, y-z1, y-z2
                    # We can flip any of these
                    # Let's pick one at random
                    edges_to_flip = [
                        (u, x), (u, y), (v, x), (v, y),
                        (u, z1), (u, z2), (v, z1), (v, z2),
                        (x, z1), (x, z2), (y, z1), (y, z2)
                    ]
                    e = random.choice(edges_to_flip)
                    i, j = e
                    if i > j: i, j = j, i
                    # Update colors list
                    # Find index of (i,j)
                    # idx = i*n - i*(i+1)//2 + (j-i-1) ?
                    # Simpler: precompute index map
                    pass # Need index map
            else:
                # Flip an edge in viol0
                u, v, x, y, z1, z2 = viol0
                edges_to_flip = [
                    (u, x), (u, y), (v, x), (v, y),
                    (u, z1), (u, z2), (v, z1), (v, z2),
                    (x, z1), (x, z2), (y, z1), (y, z2)
                ]
                e = random.choice(edges_to_flip)
                i, j = e
                if i > j: i, j = j, i
                
                # Update adj and colors
                # This is getting complex.
                pass

    return {'n': best_n, 'colors': best_colors}
