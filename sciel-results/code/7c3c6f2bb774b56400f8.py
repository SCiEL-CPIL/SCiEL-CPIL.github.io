import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute witnesses
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u,v)] = wl
            witness[(v,u)] = wl

    def get_forbidden_mask(adj, u, v):
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        mask = 0
        for x, y in wl:
            c_vx = adj[v, x]
            if c_vx == -1:
                continue
            c_xy = adj[x, y]
            if c_xy == -1:
                continue
            c_yu = adj[y, u]
            if c_yu == -1:
                continue
            if c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                return mask, True
            mask |= (1 << c_vx) | (1 << c_xy) | (1 << c_yu)
        return mask, False

    def find_valid_color(adj, u, v):
        mask, blocked = get_forbidden_mask(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if not (mask & (1 << c))]
        if not available:
            return -1
        return random.choice(available)

    def can_recolor_edge_fast(adj, u, v, new_c):
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        adj[u, v] = new_c
        adj[v, u] = new_c
        valid = True
        wl = witness[(u,v)]
        for x, y in wl:
            c_vx = adj[v, x]
            if c_vx == -1:
                continue
            c_xy = adj[x, y]
            if c_xy == -1:
                continue
            c_yu = adj[y, u]
            if c_yu == -1:
                continue
            if new_c == c_vx or new_c == c_xy or new_c == c_yu:
                valid = False
                break
        adj[u, v] = old_c
        adj[v, u] = old_c
        return valid

    def local_search(adj, max_rounds=40):
        edges_missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges_missing)
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            random.shuffle(edges_missing)
            for u, v in edges_missing:
                if adj[u, v] != -1:
                    continue
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                    edges_missing.remove((u, v))
                    progress = True
                    continue
                added = False
                for neighbor in [u, v]:
                    if added:
                        break
                    for x in range(n):
                        if x == neighbor or adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        for new_c in candidates:
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                c2 = find_valid_color(adj, u, v)
                                if c2 != -1:
                                    adj[u, v] = c2
                                    adj[v, u] = c2
                                    edges_missing.remove((u, v))
                                    added = True
                                    progress = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        if added:
                            break
            if not progress:
                break
        return int(np.sum(adj != -1) // 2)

    def try_construction(adj):
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            c = find_valid_color(adj, u, v)
            if c != -1:
                adj[u, v] = c
                adj[v, u] = c
        return local_search(adj)

    for restart in range(2000):
        if time.time() > deadline:
            break
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        if restart % 3 == 0:
            # Random spanning tree
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(1, n):
                u = vertices[i]
                v = vertices[random.randint(0, i-1)]
                c = random.randint(0, n-1)
                adj[u, v] = c
                adj[v, u] = c
        elif restart % 3 == 1:
            # Cycle + random chords
            for i in range(n):
                u, v = i, (i+1) % n
                c = random.randint(0, n-1)
                adj[u, v] = c
                adj[v, u] = c
            for _ in range(8):
                u, v = random.randint(0,n-1), random.randint(0,n-1)
                if u != v and adj[u,v] == -1:
                    c = random.randint(0, n-1)
                    adj[u, v] = c
                    adj[v, u] = c
        else:
            # Random sparse graph
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        c = random.randint(0, n-1)
                        adj[i, j] = c
                        adj[j, i] = c
        
        cnt = try_construction(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        if best_edges > 170:
            break
    
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
