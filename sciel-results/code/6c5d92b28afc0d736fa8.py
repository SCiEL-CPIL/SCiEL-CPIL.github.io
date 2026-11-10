import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def is_valid(adj, deg):
        deg_map = {}
        for i in range(n):
            d = deg[i]
            if d not in deg_map:
                deg_map[d] = []
            deg_map[d].append(i)
        
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s = verts[i]
                    t = verts[j]
                    found = False
                    for u in range(n):
                        if u == s or u == t or not adj[s][u]:
                            continue
                        for v in range(n):
                            if v == s or v == t or v == u or not adj[t][v]:
                                continue
                            if adj[u][v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Start with K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    best_adj = [[False] * n for _ in range(n)]
    for u in part_a:
        for v in part_b:
            best_adj[u][v] = True
            best_adj[v][u] = True
    
    best_deg = get_degrees(best_adj)
    best_edges = count_edges(best_adj)
    best_valid = is_valid(best_adj, best_deg)
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Greedy Add with multiple random orderings
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        random.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if best_adj[u][v]:
                continue
            best_adj[u][v] = True
            best_adj[v][u] = True
            new_deg = get_degrees(best_adj)
            if is_valid(best_adj, new_deg):
                best_deg = new_deg
                best_edges = count_edges(best_adj)
                best_valid = True
                improved = True
            else:
                best_adj[u][v] = False
                best_adj[v][u] = False
    
    # 2-opt style: remove one edge, add another (net zero), then try to add more
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        random.shuffle(all_pairs)
        for i, (u1, v1) in enumerate(all_pairs):
            if time.time() > deadline:
                break
            if not best_adj[u1][v1]:
                continue
            for (u2, v2) in all_pairs:
                if time.time() > deadline:
                    break
                if best_adj[u2][v2] or (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                best_adj[u1][v1] = False
                best_adj[v1][u1] = False
                best_adj[u2][v2] = True
                best_adj[v2][u2] = True
                new_deg = get_degrees(best_adj)
                if is_valid(best_adj, new_deg):
                    best_deg = new_deg
                    best_edges = count_edges(best_adj)
                    best_valid = True
                    improved = True
                    # Try to add edges after swap
                    for (u3, v3) in all_pairs:
                        if time.time() > deadline:
                            break
                        if best_adj[u3][v3]:
                            continue
                        best_adj[u3][v3] = True
                        best_adj[v3][u3] = True
                        nd2 = get_degrees(best_adj)
                        if is_valid(best_adj, nd2):
                            best_deg = nd2
                            best_edges = count_edges(best_adj)
                        else:
                            best_adj[u3][v3] = False
                            best_adj[v3][u3] = False
                    break
                else:
                    best_adj[u1][v1] = True
                    best_adj[v1][u1] = True
                    best_adj[u2][v2] = False
                    best_adj[v2][u2] = False
        if not improved:
            break
    
    # If still invalid, random restart
    if not best_valid:
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            adj = [[False] * n for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.4:
                        adj[u][v] = True
                        adj[v][u] = True
            deg = get_degrees(adj)
            if is_valid(adj, deg):
                best_adj = adj
                best_deg = deg
                best_edges = count_edges(adj)
                best_valid = True
                break
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
