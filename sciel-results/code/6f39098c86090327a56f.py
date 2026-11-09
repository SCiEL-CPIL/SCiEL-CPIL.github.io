import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110
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

    def get_forbidden_and_blocked(adj, u, v):
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        forbidden = set()
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
                return forbidden, True
            
            forbidden.add(c_vx)
            forbidden.add(c_xy)
            forbidden.add(c_yu)
        
        return forbidden, False

    def find_valid_color(adj, u, v):
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
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

    def local_search_polish(adj, time_limit=5.0):
        """Try to add edges or recolor to improve the solution"""
        start_time = time.time()
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges)
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for u, v in edges:
                if adj[u, v] != -1:
                    continue
                
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                    improved = True
                    continue
                
                # Try recoloring one adjacent edge to free up a color
                for neighbor in [u, v]:
                    if improved:
                        break
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
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
                                    improved = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        
                        if improved:
                            break
                    if improved:
                        break
        
        return adj

    for restart in range(1000):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Better initial construction: try to build a denser starting graph
        # Use a greedy approach with multiple orderings
        best_initial = None
        best_initial_edges = 0
        
        for init_attempt in range(3):
            adj_try = -np.ones((n, n), dtype=int)
            np.fill_diagonal(adj_try, -1)
            
            vertices = list(range(n))
            random.shuffle(vertices)
            
            # First pass: add edges greedily
            for i in range(1, n):
                u = vertices[i]
                # Try connecting to multiple previous vertices
                candidates = list(range(i))
                random.shuffle(candidates)
                added = False
                for j in candidates:
                    v = vertices[j]
                    c = find_valid_color(adj_try, u, v)
                    if c != -1:
                        adj_try[u, v] = c
                        adj_try[v, u] = c
                        added = True
                        if random.random() < 0.3:  # Sometimes stop after one edge
                            break
                if not added:
                    # At least connect to one previous vertex
                    v = vertices[random.randint(0, i-1)]
                    c = random.randint(0, n-1)
                    adj_try[u, v] = c
                    adj_try[v, u] = c
            
            # Second pass: try to add more edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if adj_try[u, v] != -1:
                    continue
                c = find_valid_color(adj_try, u, v)
                if c != -1:
                    adj_try[u, v] = c
                    adj_try[v, u] = c
            
            cnt = np.sum(adj_try != -1) // 2
            if cnt > best_initial_edges:
                best_initial_edges = cnt
                best_initial = adj_try.copy()
        
        if best_initial is not None:
            adj = best_initial
        
        # Main edge addition loop
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
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
            else:
                added = False
                for neighbor in [u, v]:
                    if added:
                        break
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
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
                                    added = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        
                        if added:
                            break
        
        # Local search polish
        if time.time() < deadline - 10:
            adj = local_search_polish(adj, time_limit=3.0)
        
        cnt = np.sum(adj != -1) // 2
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
