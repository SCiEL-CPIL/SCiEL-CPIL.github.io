import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def get_violation(n, colors):
        # Returns (u, v, c) if a mono K2,2,2 is found, else None
        adj = [[[] for _ in range(n)] for _ in range(n)]
        edge_index = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index]
                adj[i][j].append((c, j))
                adj[j][i].append((c, i))
                edge_index += 1
        
        for c in [0, 1]:
            for u in range(n):
                for (cc, v) in adj[u]:
                    if cc != c or u > v:
                        continue
                    neighbors_u = [w for (c1, w) in adj[u] if c1 == c]
                    neighbors_v = [w for (c1, w) in adj[v] if c1 == c]
                    common = set(neighbors_u).intersection(neighbors_v)
                    if len(common) < 4:
                        continue
                    common_list = list(common)
                    for i1 in range(len(common_list)):
                        b1 = common_list[i1]
                        for i2 in range(i1 + 1, len(common_list)):
                            b2 = common_list[i2]
                            # Check if b1, b2 non-adjacent in c
                            if any(w == b2 for (c1, w) in adj[b1] if c1 == c):
                                continue
                            candidates = set()
                            for w in common_list:
                                if w == b1 or w == b2:
                                    continue
                                conn_b1 = any(w2 == b1 for (c1, w2) in adj[w] if c1 == c)
                                conn_b2 = any(w2 == b2 for (c1, w2) in adj[w] if c1 == c)
                                if conn_b1 and conn_b2:
                                    candidates.add(w)
                            if len(candidates) < 2:
                                continue
                            cand_list = list(candidates)
                            for j1 in range(len(cand_list)):
                                c1 = cand_list[j1]
                                for j2 in range(j1 + 1, len(cand_list)):
                                    c2 = cand_list[j2]
                                    if any(w == c2 for (c1, w) in adj[c1] if c1 == c):
                                        continue
                                    return (u, v, c)
        return None

    def flip_edge(n, colors, u, v):
        # Find index
        # i < j
        if u > v:
            u, v = v, u
        # index = sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
        # index = u*n - u*(u+1)//2 + v - u - 1
        idx = u * n - u * (u + 1) // 2 + (v - u - 1)
        colors[idx] = 1 - colors[idx]

    n_try = 29
    squares = {x * x % n_try for x in range(1, n_try)}
    colors = [0 if (j - i) % n_try in squares else 1 for i in range(n_try) for j in range(i + 1, n_try)]
    
    # Check and repair
    for _ in range(1000):
        if time.time() > deadline:
            break
        viol = get_violation(n_try, colors)
        if viol is None:
            return {'n': n_try, 'colors': colors}
        u, v, c = viol
        # Flip an edge in the violation
        flip_edge(n_try, colors, u, v)
        
    # Fallback to 17
    n_fallback = 17
    squares = {x * x % n_fallback for x in range(1, n_fallback)}
    colors = [0 if (j - i) % n_fallback in squares else 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    return {'n': n_fallback, 'colors': colors}
