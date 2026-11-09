import time
import random

def _grid_adj(a1, b1, a2, b2):
    da = abs(a1 - a2)
    da = min(da, 5 - da)
    db = abs(b1 - b2)
    db = min(db, 10 - db)
    if a1 == a2:
        return db == 1
    if b1 == b2:
        return da == 1
    return False

def _neighbors_grid(N):
    adj = [[] for _ in range(N)]
    for a in range(5):
        for b in range(10):
            u = a * 10 + b
            for a2 in range(5):
                for b2 in range(10):
                    if (a, b) != (a2, b2) and _grid_adj(a, b, a2, b2):
                        adj[u].append(a2 * 10 + b2)
    return adj

def _proper_4color(adj, N, deadline, seed=0):
    rng = random.Random(seed)
    order = list(range(N))
    order.sort(key=lambda v: -len(adj[v]))
    for i in range(N - 1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
    
    color = [-1] * N
    
    def assign(pos):
        if time.time() > deadline:
            return False
        if pos == N:
            return True
        v = order[pos]
        used = set()
        for nb in adj[v]:
            if color[nb] != -1:
                used.add(color[nb])
        colors_to_try = list(range(4))
        rng.shuffle(colors_to_try)
        for c in colors_to_try:
            if c not in used:
                color[v] = c
                if assign(pos + 1):
                    return True
                color[v] = -1
        return False
    
    if assign(0):
        return color
    return None

def _build():
    N = 50
    adj = _neighbors_grid(N)
    adj_set = [set(nei) for nei in adj]
    comp_adj = []
    for i in range(N):
        comp_nei = [j for j in range(N) if j != i and j not in adj_set[i]]
        comp_adj.append(comp_nei)
    
    deadline = time.time() + 100.0
    
    for seed in range(100):
        if time.time() > deadline:
            break
        cg = _proper_4color(adj, N, deadline, seed)
        if cg is None:
            continue
        cc = _proper_4color(comp_adj, N, deadline, seed + 100)
        if cc is None:
            continue
        
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if j in adj_set[i]:
                    colors.append(cg[i])
                else:
                    colors.append(cc[i])
        
        if _verify(N, colors):
            return N, colors
            
    return None

def _verify(n, colors):
    N = n
    col = [[0] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            col[i][j] = col[j][i] = colors[k]
            k += 1
    for i in range(N):
        for j in range(i + 1, N):
            for l in range(j + 1, N):
                if col[i][j] == col[j][l] == col[i][l]:
                    return False
    return True

def _seed32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    c = []
    for i in range(32):
        for j in range(i + 1, 32):
            c.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': c}

def construct():
    deadline = time.time() + 140.0
    r = _build()
    if r is not None and time.time() < deadline:
        n, colors = r
        return {'n': n, 'colors': colors}
    return _seed32()
