import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

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
        # Check all 3-paths v-w-x-u
        # A 4-cycle is formed if we add (u,v) and there exists a path u-x-w-v of length 3.
        # The edges are (u,v), (v,w), (w,x), (x,u).
        # Colors must be distinct: c_uv != c_vw != c_wx != c_xu != c_uv.
        # Since c_uv is new, we just need to ensure c_uv is not in {c_vw, c_wx, c_xu}
        # AND that {c_vw, c_wx, c_xu} are distinct among themselves.
        
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
                    # If the existing 3-path has duplicate colors, 
                    # adding (u,v) creates a 4-cycle with duplicate colors regardless of c_uv.
                    # Thus, we cannot add (u,v) at all.
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
            # Sort by sum of indices as a proxy for "central" edges first?
            # Or just random.
            all_edges.sort(key=lambda e: (-(e[0] + e[1]), e))
        elif order == 'far':
            # Sort by distance in some metric?
            all_edges.sort(key=lambda e: (abs(e[0] - e[1]), e))
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors(u, v, adj, color_mat, neighbors)
            if valid:
                valid_list = list(valid)
                if prefer_formula:
                    # Try a few formulas
                    c1 = (u + v) % num_colors
                    c2 = (u * v) % num_colors
                    c3 = (u ^ v) % num_colors
                    
                    # Prefer formula if valid, else random
                    chosen = None
                    if c1 in valid:
                        chosen = c1
                    elif c2 in valid:
                        chosen = c2
                    elif c3 in valid:
                        chosen = c3
                    else:
                        chosen = random.choice(valid_list)
                    c = chosen
                else:
                    c = random.choice(valid_list)
                
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
        
        def get_valid_colors_local(u, v):
            # Same logic, but uses local variables
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

        # Track colors used to potentially balance?
        color_counts = [0] * num_colors
        for e in cur_edges:
            color_counts[e[2]] += 1

        for iteration in range(200):
            if not time_ok():
                break
            
            # Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            if not missing:
                break
            random.shuffle(missing)
            
            added = False
            for u, v in missing:
                valid = get_valid_colors_local(u, v)
                if valid:
                    # Choose color that is least frequent?
                    valid_list = list(valid)
                    # Sort by count
                    valid_list.sort(key=lambda c: color_counts[c])
                    c = valid_list[0]
                    
                    adj[u][v] = True
                    adj[v][u] = True
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    cur_edges.append([u, v, c])
                    color_counts[c] += 1
                    cur_count += 1
                    edge_list.append((u, v))
                    added = True
                    break
            
            if not added:
                # Try recoloring
                improved = False
                random.shuffle(edge_list)
                # Try to recolor edges that are involved in many potential 4-cycles?
                # Just random for now.
                for eu, ev in edge_list[:50]:
                    old_c = color_mat[eu][ev]
                    other_colors = [c for c in range(num_colors) if c != old_c]
                    random.shuffle(other_colors)
                    
                    for new_c in other_colors[:10]:
                        # Check if recoloring (eu, ev) to new_c is valid
                        valid_recolor = True
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
                                
                                # The existing 3-path colors must be distinct
                                if c_ev_w == c_wx or c_ev_w == c_xu or c_wx == c_xu:
                                    # This path is already bad? 
                                    # If so, current graph is invalid. 
                                    # But we assume current graph is valid.
                                    pass 
                                    
                                # Check if new_c clashes
                                if new_c in (c_ev_w, c_wx, c_xu):
                                    valid_recolor = False
                                    break
                            if not valid_recolor:
                                break
                        
                        if valid_recolor:
                            # Perform recolor
                            color_mat[eu][ev] = new_c
                            color_mat[ev][eu] = new_c
                            color_counts[old_c] -= 1
                            color_counts[new_c] += 1
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
    strategies = [
        (True, 'random'),
        (False, 'random'),
        (True, 'bipartite'),
        (False, 'bipartite'),
        (True, 'degree'),
        (False, 'degree'),
        (True, 'far'),
        (False, 'far'),
    ]
    
    # Run multiple restarts
    num_restarts = 1000
    
    for strategy_idx, (prefer_formula, order) in enumerate(strategies):
        for seed in range(num_restarts):
            if not time_ok():
                break
            edges = run_greedy(seed + strategy_idx * 10000, prefer_formula=prefer_formula, order=order)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [e[:] for e in edges]
            
            # If we found a very good solution, run local search on it
            if best_count > 80 and time_ok():
                improved = local_search(best_edges, seed)
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_edges = [e[:] for e in improved]
    
    # Final local search on the best found
    if best_edges and time_ok():
        improved = local_search(best_edges, 99999)
        if len(improved) > best_count:
            best_count = len(improved)
            best_edges = [e[:] for e in improved]

    return best_edges
