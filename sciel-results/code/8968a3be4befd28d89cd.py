import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 117
    
    def time_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = 0
    
    def run_greedy(seed, prefer_formula=True, order='random'):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        def get_valid(u, v):
            forbidden = set()
            for w in neighbors[v]:
                if w == u:
                    continue
                c_vw = color_mat[v][w]
                for x in neighbors[u]:
                    if x == v or x == w:
                        continue
                    if not adj[w][x]:
                        continue
                    c_wx = color_mat[w][x]
                    c_xu = color_mat[x][u]
                    if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                        return set()
                    forbidden.add(c_vw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xu)
            return set(range(num_colors)) - forbidden
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        if order == 'random':
            random.shuffle(all_edges)
        elif order == 'bipartite':
            # Try bipartite-like ordering: alternate between two halves
            left = list(range(10))
            right = list(range(10, 20))
            bip_edges = [(u, v) for u in left for v in right]
            intra_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (i < 10 and j >= 10)]
            random.shuffle(bip_edges)
            random.shuffle(intra_edges)
            all_edges = bip_edges + intra_edges
        elif order == 'degree':
            # Order by potential degree (vertices with more potential neighbors first)
            all_edges.sort(key=lambda e: (-(e[0] + e[1]), e))
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid(u, v)
            if valid:
                if prefer_formula:
                    pref = (u + v) % num_colors
                    if pref in valid:
                        c = pref
                    else:
                        c = random.choice(list(valid))
                else:
                    c = random.choice(list(valid))
                adj[u][v] = True
                adj[v][u] = True
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        return edges
    
    def local_search(edges, seed):
        """Try to add more edges via recoloring."""
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edge_list = []
        
        for u, v, c in edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_list.append((u, v))
        
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        def get_valid(u, v):
            forbidden = set()
            for w in neighbors[v]:
                if w == u:
                    continue
                c_vw = color_mat[v][w]
                for x in neighbors[u]:
                    if x == v or x == w:
                        continue
                    if not adj[w][x]:
                        continue
                    c_wx = color_mat[w][x]
                    c_xu = color_mat[x][u]
                    if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                        return set()
                    forbidden.add(c_vw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xu)
            return set(range(num_colors)) - forbidden
        
        for iteration in range(300):
            if not time_ok():
                break
            
            # Try adding missing edges - try all missing edges, not just first 100
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing:
                valid = get_valid(u, v)
                if valid:
                    c = random.choice(list(valid))
                    adj[u][v] = True
                    adj[v][u] = True
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    cur_edges.append([u, v, c])
                    cur_count += 1
                    edge_list.append((u, v))
                    added = True
                    break
            
            if not added:
                # Try recoloring - try more edges and more color options
                improved = False
                random.shuffle(edge_list)
                for eu, ev in edge_list[:80]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    for new_c in other_colors[:15]:
                        # Check validity
                        valid = True
                        for w in neighbors[ev]:
                            if w == eu:
                                continue
                            c_ev_w = color_mat[ev][w]
                            for x in neighbors[eu]:
                                if x == ev or x == w:
                                    continue
                                if not adj[w][x]:
                                    continue
                                c_wx = color_mat[w][x]
                                c_xu = color_mat[x][eu]
                                if new_c in (c_ev_w, c_wx, c_xu):
                                    valid = False
                                    break
                                if c_ev_w in (c_wx, c_xu) or c_wx == c_xu:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        color_mat[eu][ev] = new_c
                        color_mat[ev][eu] = new_c
                        for i, e in enumerate(cur_edges):
                            if e[0] == eu and e[1] == ev:
                                e[2] = new_c
                                break
                        improved = True
                        break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return cur_edges
    
    # Phase 1: Greedy with multiple strategies
    # Try different edge orderings and formula preferences
    strategies = [
        (True, 'random'),
        (False, 'random'),
        (True, 'bipartite'),
        (False, 'bipartite'),
        (True, 'degree'),
    ]
    
    for strategy_idx, (prefer_formula, order) in enumerate(strategies):
        for seed in range(500):
            if not time_ok():
                break
            edges = run_greedy(seed + strategy_idx * 10000, prefer_formula=prefer_formula, order=order)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [e[:] for e in edges]
    
    # Phase 2: Local search on best solutions - try more iterations
    if best_edges and time_ok():
        for seed in range(300):
            if not time_ok():
                break
            improved = local_search(best_edges, seed)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = [e[:] for e in improved]
    
    return best_edges
