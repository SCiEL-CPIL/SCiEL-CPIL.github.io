import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    
    def check(colors):
        adj = [[-1]*n for _ in range(n)]
        for idx, (i, j) in enumerate(edges):
            adj[i][j] = adj[j][i] = colors[idx]
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        ec = {adj[a][b], adj[a][c], adj[a][d], adj[b][c], adj[b][d], adj[c][d]}
                        if len(ec) < 5:
                            violations += 1
        return violations, len(set(colors))
    
    candidates = []
    
    for p in range(13, 21):
        candidates.append([(i+j) % p for i, j in edges])
        candidates.append([(i + 2*j) % p for i, j in edges])
        candidates.append([(2*i + j) % p for i, j in edges])
        candidates.append([(i * j) % p for i, j in edges])
        candidates.append([(i*i + j*j) % p for i, j in edges])
        candidates.append([(i*i - j*j) % p for i, j in edges])
        candidates.append([(i+j+1) % p for i, j in edges])
        candidates.append([(i*j + i + j) % p for i, j in edges])
    
    for p in range(13, 21):
        for alpha in range(1, 5):
            for beta in range(1, 5):
                cols = []
                for i, j in edges:
                    a, b = divmod(i, 4)
                    c, d = divmod(j, 4)
                    cols.append((alpha*(a-c) + beta*(b-d)) % p)
                candidates.append(cols)
        cols = []
        for i, j in edges:
            a, b = divmod(i, 4)
            c, d = divmod(j, 4)
            cols.append((a*c + b*d) % p)
        candidates.append(cols)
        cols = []
        for i, j in edges:
            a, b = divmod(i, 4)
            c, d = divmod(j, 4)
            cols.append((a*d + b*c) % p)
        candidates.append(cols)
    
    best = None
    best_nc = 999
    for cols in candidates:
        viol, nc = check(cols)
        if viol == 0 and nc < best_nc:
            best_nc = nc
            best = cols[:]
    
    if best is None:
        best = [(i+j) % 17 for i, j in edges]
        best_nc = 17
    
    # Local search
    cur = best[:]
    cur_viol, cur_nc = check(cur)
    
    for _ in range(50000):
        if time.time() > deadline - 5:
            break
        e = random.randint(0, num_edges - 1)
        old = cur[e]
        new = random.randint(0, 15)
        if new == old:
            continue
        cur[e] = new
        v, nc = check(cur)
        if v < cur_viol or (v == cur_viol and nc < cur_nc):
            cur_viol, cur_nc = v, nc
            if v == 0 and nc < best_nc:
                best_nc = nc
                best = cur[:]
        else:
            cur[e] = old
    
    result = best
    max_c = max(result)
    if max_c > 119:
        color_map = {}
        new_result = []
        next_c = 0
        for c in result:
            if c not in color_map:
                color_map[c] = next_c
                next_c += 1
            new_result.append(color_map[c])
        result = new_result
    return result
