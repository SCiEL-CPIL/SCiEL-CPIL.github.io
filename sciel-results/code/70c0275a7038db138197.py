import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit

    def time_ok():
        return time.time() < deadline

    best_edges = []
    best_count = 0

    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns the set of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property.
        Assumes (u, v) is NOT currently in the graph.
        """
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
                
                # Check if existing path colors are distinct
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return set()
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        
        return set(range(num_colors)) - forbidden

    def run_greedy(seed, prefer_formula=True, order='random'):
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        if order == 'random':
            random.shuffle(all_edges)
        elif order == 'bipartite':
            left = list(range(10))
            right = list(range(10, 20))
            bip_edges = [(u, v) for u in left for v in right]
            intra_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (i < 10 and j >= 10)]
            random.shuffle(bip_edges)
            random.shuffle(intra_edges)
            all_edges = bip_edges + intra_edges
        elif order == 'degree':
            # Sort by sum of vertex indices (lower sum = more "central" in some sense)
            all_edges.sort(key=lambda e: (e[0] + e[1], e))
        elif order == 'degree_desc':
            all_edges.sort(key=lambda e: (-(e[0] + e[1]), e))
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors(u, v, adj, color_mat, neighbors)
            if valid:
                if prefer_formula:
                    # Try multiple formula-based color choices
                    candidates = [(u + v) % num_colors, (u * 3 + v) % num_colors, (u + 2*v) % num_colors]
                    c = None
                    for pref in candidates:
                        if pref in valid:
                            c = pref
                            break
                    if c is None:
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
        """Try to add more edges via recoloring and edge swaps."""
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        edge_list = []
        
        for u, v, c in edges:
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
            edge_list.append((u, v))
        
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        def get_valid(u, v):
            return get_valid_colors(u, v, adj, color_mat, neighbors)
        
        def is_valid_recolor(eu, ev, new_c):
            """Check if recoloring edge (eu, ev) to new_c is valid."""
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
                    
                    # New color must not clash with path colors
                    if new_c == c_ev_w or new_c == c_wx or new_c == c_xu:
                        return False
                    # Path colors must be distinct
                    if c_ev_w == c_wx or c_ev_w == c_xu or c_wx == c_xu:
                        return False
            return True
        
        def perform_recolor(eu, ev, new_c):
            color_mat[eu][ev] = new_c
            color_mat[ev][eu] = new_c
            for i, e in enumerate(cur_edges):
                if (e[0] == eu and e[1] == ev) or (e[0] == ev and e[1] == eu):
                    e[2] = new_c
                    break
        
        def add_edge(u, v, c):
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
            edge_list.append((u, v))
            cur_edges.append([u, v, c])
        
        def remove_edge(u, v):
            c = color_mat[u][v]
            adj[u][v] = False
            adj[v][u] = False
            color_mat[u][v] = -1
            color_mat[v][u] = -1
            neighbors[u].discard(v)
            neighbors[v].discard(u)
            edge_set.discard((min(u,v), max(u,v)))
            edge_list.remove((u, v))
            for i, e in enumerate(cur_edges):
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    cur_edges.pop(i)
                    break
        
        for iteration in range(1000):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing:
                valid = get_valid(u, v)
                if valid:
                    c = random.choice(list(valid))
                    add_edge(u, v, c)
                    cur_count += 1
                    added = True
                    break
            
            if not added:
                # Try recoloring to enable adding an edge
                improved = False
                
                # Strategy 1: Single edge recolor
                random.shuffle(edge_list)
                for eu, ev in edge_list[:150]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    
                    for new_c in other_colors[:25]:
                        if is_valid_recolor(eu, ev, new_c):
                            # Check if this recolor enables adding any missing edge
                            perform_recolor(eu, ev, new_c)
                            
                            # Try to add an edge
                            added_after_recolor = False
                            for mu, mv in missing[:50]:
                                valid_after = get_valid(mu, mv)
                                if valid_after:
                                    c_new = random.choice(list(valid_after))
                                    add_edge(mu, mv, c_new)
                                    cur_count += 1
                                    added_after_recolor = True
                                    break
                            
                            if not added_after_recolor:
                                # Revert if no edge was added
                                remove_edge(mu, mv) if added_after_recolor else None
                                perform_recolor(eu, ev, old_c)
                            else:
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    continue
                
                # Strategy 2: Two-edge swap (remove one, try to add another)
                random.shuffle(edge_list)
                for eu, ev in edge_list[:80]:
                    old_c = color_mat[eu][ev]
                    remove_edge(eu, ev)
                    cur_count -= 1
                    
                    # Try to add a different edge
                    added_swap = False
                    for mu, mv in missing[:30]:
                        if (mu, mv) == (eu, ev) or (mu, mv) == (ev, eu):
                            continue
                        valid_swap = get_valid(mu, mv)
                        if valid_swap:
                            c_swap = random.choice(list(valid_swap))
                            add_edge(mu, mv, c_swap)
                            cur_count += 1
                            added_swap = True
                            break
                    
                    if not added_swap:
                        # Revert: re-add the original edge
                        add_edge(eu, ev, old_c)
                        cur_count += 1
                
                if not improved and not added_swap:
                    # Try more aggressive: multiple recolors
                    multi_improved = False
                    for _ in range(10):
                        random.shuffle(edge_list)
                        for eu, ev in edge_list[:50]:
                            old_c = color_mat[eu][ev]
                            other_colors = [c for c in range(num_colors) if c != old_c]
                            random.shuffle(other_colors)
                            for new_c in other_colors[:10]:
                                if is_valid_recolor(eu, ev, new_c):
                                    perform_recolor(eu, ev, new_c)
                                    # Check if any edge can now be added
                                    for mu, mv in missing[:20]:
                                        valid_check = get_valid(mu, mv)
                                        if valid_check:
                                            c_add = random.choice(list(valid_check))
                                            add_edge(mu, mv, c_add)
                                            cur_count += 1
                                            multi_improved = True
                                            break
                                    if multi_improved:
                                        break
                            if multi_improved:
                                break
                        if multi_improved:
                            break
                    if multi_improved:
                        continue
                    else:
                        break
        
        return cur_edges

    # Phase 1: Greedy with multiple strategies
    strategies = [
        (True, 'random'),
        (False, 'random'),
        (True, 'bipartite'),
        (False, 'bipartite'),
        (True, 'degree'),
        (False, 'degree'),
        (True, 'degree_desc'),
        (False, 'degree_desc'),
    ]
    
    # Use more seeds for better coverage
    seeds_per_strategy = 400
    
    for strategy_idx, (prefer_formula, order) in enumerate(strategies):
        for seed in range(seeds_per_strategy):
            if not time_ok():
                break
            edges = run_greedy(seed + strategy_idx * 100000, prefer_formula=prefer_formula, order=order)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [e[:] for e in edges]
    
    # Phase 2: Local search on best solutions
    if best_edges and time_ok():
        for seed in range(300):
            if not time_ok():
                break
            improved = local_search(best_edges, seed)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = [e[:] for e in improved]
    
    return best_edges
