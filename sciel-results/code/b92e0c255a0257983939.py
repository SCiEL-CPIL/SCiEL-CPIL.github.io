import numpy as np
import time
import random

def construct():
    n = 20
    best_edges = 0
    best_adj = None
    
    deadline = time.time() + 115
    
    random.seed(42)
    
    def count_edges(adj):
        return int(np.sum(adj != -1) // 2)
    
    def try_add_edge(adj, u, v, c):
        for x in range(n):
            if x == u or x == v or adj[v, x] == -1:
                continue
            c_vx = adj[v, x]
            for y in range(n):
                if y == u or y == v or y == x or adj[u, y] == -1 or adj[x, y] == -1:
                    continue
                c_xy = adj[x, y]
                c_yu = adj[y, u]
                if c == c_vx or c == c_xy or c == c_yu or c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                    return False
        adj[u, v] = c
        adj[v, u] = c
        return True
    
    def fix_invalid(adj):
        removed = 0
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    return removed
                for v in range(u+1, n):
                    if adj[u, v] == -1:
                        continue
                    cv = adj[u, v]
                    invalid = False
                    for x in range(n):
                        if x == u or x == v or adj[v, x] == -1:
                            continue
                        c_vx = adj[v, x]
                        for y in range(n):
                            if y == u or y == v or y == x or adj[u, y] == -1 or adj[x, y] == -1:
                                continue
                            c_xy = adj[x, y]
                            c_yu = adj[y, u]
                            if cv == c_vx or cv == c_xy or cv == c_yu or c_vx == c_xy or c_vx == c_yu or c_xy == c_yu:
                                invalid = True
                                break
                        if invalid:
                            break
                    if invalid:
                        adj[u, v] = -1
                        adj[v, u] = -1
                        removed += 1
                        changed = True
        return removed
    
    # Phase 1: Structured initializations
    initializations = []
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(n):
        for v in range(u+1, n):
            c = (u + v) % n
            adj[u, v] = c
            adj[v, u] = c
    initializations.append(adj.copy())
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(n):
        for v in range(u+1, n):
            c = (u - v) % n
            adj[u, v] = c
            adj[v, u] = c
    initializations.append(adj.copy())
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(n):
        for v in range(u+1, n):
            c = (u * v) % n
            if c == 0:
                c = n - 1
            adj[u, v] = c
            adj[v, u] = c
    initializations.append(adj.copy())
    
    for _ in range(5):
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in range(u+1, n):
                c = random.randint(0, n-1)
                adj[u, v] = c
                adj[v, u] = c
        initializations.append(adj.copy())
    
    for adj in initializations:
        if time.time() > deadline:
            break
        fix_invalid(adj)
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
    
    # Phase 2: Local search on best
    if best_adj is not None:
        adj = best_adj.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline:
                        break
                    for c in range(n):
                        if try_add_edge(adj, u, v, c):
                            improved = True
                            break
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj
    
    # Phase 3: Perturbation + local search
    for _ in range(100):
        if time.time() > deadline:
            break
        if best_adj is not None:
            adj = best_adj.copy()
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
            random.shuffle(edges)
            for u, v in edges[:random.randint(5, 20)]:
                adj[u, v] = -1
                adj[v, u] = -1
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u+1, n):
                    c = random.randint(0, n-1)
                    adj[u, v] = c
                    adj[v, u] = c
        
        fix_invalid(adj)
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if time.time() < deadline - 10:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline - 5:
                        break
                    for c in range(n):
                        if try_add_edge(adj, u, v, c):
                            improved = True
                            break
            cnt = count_edges(adj)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = adj.copy()
    
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
