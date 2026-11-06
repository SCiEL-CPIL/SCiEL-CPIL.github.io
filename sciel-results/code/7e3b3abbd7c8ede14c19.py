"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _edge_idx(i, j, n):
    return i * (2*n - 1 - i) // 2 + (j - i - 1)

def _build_adj(colors, n):
    adj = [[set() for _ in range(n)] for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj

def _has_c6_fast(adj, n, color):
    adj_c = adj[color]
    for u in range(n):
        for v in adj_c[u]:
            if u >= v:
                continue
            d2 = set()
            for a in adj_c[u]:
                if a == v:
                    continue
                d2.update(adj_c[a])
            d2.discard(u)
            d2.discard(v)
            for z in d2:
                if v in adj_c[z]:
                    return True
    return False

def _initial_n9():
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return n, colors

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    n, colors = _initial_n9()
    best_n = n
    best_colors = colors[:]
    
    for target_n in range(10, 21):
        if time.time() > deadline - 15:
            break
        
        result = _backtrack(base_colors=best_colors, base_n=best_n, target_n=target_n, deadline=deadline)
        if result is not None:
            best_n = target_n
            best_colors = result
        else:
            break
    
    return {'n': best_n, 'colors': best_colors}

def _backtrack(base_colors, base_n, target_n, deadline):
    adj = [[set() for _ in range(target_n)] for _ in range(4)]
    idx = 0
    for i in range(base_n):
        for j in range(i+1, base_n):
            c = base_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    
    edges = []
    for new_v in range(base_n, target_n):
        for old_v in range(new_v):
            edges.append((old_v, new_v))
    
    def bt(pos):
        if time.time() > deadline - 3:
            return False
        if pos == len(edges):
            for c in range(4):
                if _has_c6_fast(adj, target_n, c):
                    return False
            return True
        
        u, v = edges[pos]
        for c in range(4):
            adj[c][u].add(v)
            adj[c][v].add(u)
            if not _has_c6_fast(adj, target_n, c):
                if bt(pos + 1):
                    return True
            adj[c][u].discard(v)
            adj[c][v].discard(u)
        return False
    
    if bt(0):
        full = base_colors[:]
        for (u,v) in edges:
            for c in range(4):
                if v in adj[c][u]:
                    full.append(c)
                    break
        return full
    return None

# EVOLVE-BLOCK-END
