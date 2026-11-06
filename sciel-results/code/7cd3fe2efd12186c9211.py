import random
import time

def construct():
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_edges = []
    best_count = 0
    
    # Number of restarts
    # We have 120 seconds. Let's estimate time per restart.
    # 190 edges * 20 checks * 20 inner loops = 76,000 ops per restart.
    # 1000 restarts = 76,000,000 ops. Should be < 5 seconds.
    # Let's do 5000 restarts to be safe and find a better max.
    restarts = 5000
    
    for _ in range(restarts):
        adj = [[0] * n for _ in range(n)]
        edges = []
        
        # Shuffle the order of edges to try
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u, v) is valid
            # 1. Check codeg(u, v) <= 2
            # Current common neighbors of u and v
            # We can maintain a codeg matrix, or calculate on the fly.
            # Calculating on the fly is O(n).
            
            # Let's maintain a codeg matrix for speed?
            # codeg[i][j] = number of common neighbors of i and j.
            # When we add (u, v):
            # - For all x in N(u) \ {v}, if v in N(x), codeg(u, x) += 1?
            #   Actually, if we add (u, v), then v is a neighbor of u.
            #   For any x such that adj[v][x] is 1, v is a common neighbor of u and x?
            #   Yes, if adj[u][x] is also 1? No.
            #   Common neighbors of u and x are {z | adj[u][z] and adj[x][z]}.
            #   If we add (u, v), then adj[u][v] becomes 1.
            #   So v is a common neighbor of u and x IF adj[x][v] is 1.
            #   So for all x where adj[v][x] is 1 (and x != u), if adj[u][x] is 1, then codeg(u, x) increases by 1.
            #   Wait, if adj[u][x] is 0, then x is not a neighbor of u, so codeg(u, x) is 0? No, codeg is defined for any pair.
            #   If adj[u][x] is 0, then N(u) does not contain x. The common neighbors are still calculated based on N(u) and N(x).
            #   Adding (u, v) adds v to N(u).
            #   So for any x, if v is in N(x), then v is a new common neighbor of u and x?
            #   Only if v is also in N(u) (which it is, after add) AND v is in N(x).
            #   So yes, for all x such that adj[v][x] is 1, codeg(u, x) increases by 1.
            #   BUT, we must ensure that the NEW codeg(u, x) does not exceed 2.
            #   So before adding, we need: for all x in N(v), codeg(u, x) <= 1 (if x != u).
            #   Similarly, for all y in N(u), codeg(v, y) <= 1 (if y != v).
            #   And codeg(u, v) <= 2.
            
            valid = True
            
            # Check codeg(u, v)
            # Count common neighbors of u and v
            cnt_uv = 0
            for z in range(n):
                if adj[u][z] and adj[v][z]:
                    cnt_uv += 1
            if cnt_uv > 2:
                valid = False
            
            if valid:
                # Check neighbors of v
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v][x]:
                        # v is a neighbor of x. After adding (u, v), v is also neighbor of u.
                        # So v is a common neighbor of u and x.
                        # This increases codeg(u, x) by 1.
                        # So we need current codeg(u, x) <= 1.
                        cnt_ux = 0
                        for z in range(n):
                            if z == u or z == x:
                                continue
                            if adj[u][z] and adj[x][z]:
                                cnt_ux += 1
                        if cnt_ux > 1:
                            valid = False
                            break
            
            if valid:
                # Check neighbors of u
                for y in range(n):
                    if y == u or y == v:
                        continue
                    if adj[u][y]:
                        # u is a neighbor of y. After adding (u, v), u is also neighbor of v.
                        # So u is a common neighbor of v and y.
                        # This increases codeg(v, y) by 1.
                        # So we need current codeg(v, y) <= 1.
                        cnt_vy = 0
                        for z in range(n):
                            if z == v or z == y:
                                continue
                            if adj[v][z] and adj[y][z]:
                                cnt_vy += 1
                        if cnt_vy > 1:
                            valid = False
                            break
            
            if valid:
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges
