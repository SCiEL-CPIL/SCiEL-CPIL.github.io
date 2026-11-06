import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    best_edges = []
    best_count = 0
    
    def get_valid_colors(u, v, adj, color_mat, neighbors):
        """
        Returns bitmask of colors that can be assigned to edge (u, v)
        without violating the rainbow 4-cycle property.
        Assumes (u, v) is NOT currently in the graph.
        """
        forbidden = 0  # bitmask
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
                # The 3 edges of the path are c_vw, c_wx, c_xu
                # If any two are equal, the cycle cannot be rainbow regardless of c_uv
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return 0
                # Otherwise, c_uv must be different from these 3
                forbidden |= (1 << c_vw) | (1 << c_wx) | (1 << c_xu)
        
        return ((1 << num_colors) - 1) & ~forbidden

    def is_valid_recolor(u, v, new_c, adj, color_mat, neighbors):
        """
        Checks if recoloring edge (u, v) to new_c is valid.
        (u, v) is currently in the graph.
        """
        old_c = color_mat[u][v]
        if new_c == old_c:
            return True
        
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
                
                # The 4 edges are (u,v), (v,w), (w,x), (x,u)
                # Colors: new_c, c_vw, c_wx, c_xu
                # They must be all distinct.
                if new_c == c_vw or new_c == c_wx or new_c == c_xu:
                    return False
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return False
        
        return True

    def run_greedy_init(seed, order='bipartite', prefer_formula=True):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        if order == 'bipartite':
            left = list(range(10))
            right = list(range(10, 20))
            bip_edges = [(u, v) for u in left for v in right]
            intra_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (i < 10 and j >= 10)]
            random.shuffle(bip_edges)
            random.shuffle(intra_edges)
            all_edges = bip_edges + intra_edges
        elif order == 'random':
            random.shuffle(all_edges)
        else:
            random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not time_ok():
                break
            valid = get_valid_colors(u, v, adj, color_mat, neighbors)
            if valid:
                if prefer_formula:
                    pref = (u + v) % num_colors
                    if (valid >> pref) & 1:
                        c = pref
                    else:
                        valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                        c = random.choice(valid_list)
                else:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                adj[u][v] = True
                adj[v][u] = True
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        return adj, color_mat, neighbors, edges

    def local_search(adj, color_mat, neighbors, edges, seed):
        random.seed(seed)
        cur_edges = [e[:] for e in edges]
        cur_count = len(cur_edges)
        
        def add_edge(u, v, c):
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            cur_edges.append([u, v, c])
            nonlocal cur_count
            cur_count += 1

        def remove_edge(u, v):
            adj[u][v] = False
            adj[v][u] = False
            color_mat[u][v] = -1
            color_mat[v][u] = -1
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            for i, e in enumerate(cur_edges):
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    cur_edges.pop(i)
                    break
            nonlocal cur_count
            cur_count -= 1

        def recolor_edge(u, v, new_c):
            color_mat[u][v] = new_c
            color_mat[v][u] = new_c
            for e in cur_edges:
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    e[2] = new_c
                    break

        def try_add_any_edge():
            """Try to add any missing edge. Returns True if successful."""
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(missing)
            for u, v in missing:
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                    add_edge(u, v, c)
                    return True
            return False

        def try_recolor_any_edge():
            """Try to recolor any existing edge. Returns True if successful."""
            if not cur_edges:
                return False
            # Try more random edges with more color attempts
            attempts = min(len(cur_edges), 25)
            indices = random.sample(range(len(cur_edges)), attempts)
            for idx in indices:
                eu, ev, ec = cur_edges[idx]
                other_colors = [c for c in range(num_colors) if c != ec]
                random.shuffle(other_colors)
                for new_c in other_colors[:15]:  # Try up to 15 colors per edge
                    if is_valid_recolor(eu, ev, new_c, adj, color_mat, neighbors):
                        recolor_edge(eu, ev, new_c)
                        return True
            return False

        def try_edge_swap():
            """Try to swap: remove one edge, then add another. Returns True if net gain."""
            if not cur_edges:
                return False
            # Try up to 3 random edges to remove
            for _ in range(3):
                idx = random.randint(0, len(cur_edges) - 1)
                eu, ev, ec = cur_edges[idx]
                
                # Temporarily remove
                remove_edge(eu, ev)
                
                # Try to add a different edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
                random.shuffle(missing)
                added = False
                for u, v in missing:
                    valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                    if valid:
                        valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                        c = random.choice(valid_list)
                        add_edge(u, v, c)
                        added = True
                        break
                
                if added:
                    # Try to add more edges after the swap
                    for _ in range(3):
                        if not try_add_any_edge():
                            break
                    return True
                else:
                    # Re-add the original edge
                    add_edge(eu, ev, ec)
            
            return False

        max_no_progress = 50
        no_progress = 0
        
        while time_ok():
            # Phase 1: Try to add edges
            if try_add_any_edge():
                no_progress = 0
                continue
            
            # Phase 2: Try to recolor
            if try_recolor_any_edge():
                no_progress += 1
                if no_progress >= max_no_progress:
                    break
                continue
            
            # Phase 3: Try edge swap
            if try_edge_swap():
                no_progress = 0
                continue
            
            # Stuck
            break

        return cur_edges

    def final_polish(adj, color_mat, neighbors, edges):
        """Final high-precision polish pass to squeeze out remaining edges."""
        cur_edges = [e[:] for e in edges]
        
        def add_edge(u, v, c):
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            cur_edges.append([u, v, c])

        def recolor_edge(u, v, new_c):
            color_mat[u][v] = new_c
            color_mat[v][u] = new_c
            for e in cur_edges:
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    e[2] = new_c
                    break

        # Multiple rounds of trying to add edges with recoloring assistance
        for round_num in range(5):
            if not time_ok():
                break
            
            # Try all missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            for u, v in missing:
                if not time_ok():
                    break
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                    add_edge(u, v, c)
            
            # Try recoloring to enable more additions
            for _ in range(10):
                if not time_ok():
                    break
                improved = False
                for e in cur_edges[:]:
                    eu, ev, ec = e
                    other_colors = [c for c in range(num_colors) if c != ec]
                    random.shuffle(other_colors)
                    for new_c in other_colors[:10]:
                        if is_valid_recolor(eu, ev, new_c, adj, color_mat, neighbors):
                            recolor_edge(eu, ev, new_c)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            # Try adding edges again after recoloring
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            for u, v in missing:
                if not time_ok():
                    break
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    valid_list = [c for c in range(num_colors) if (valid >> c) & 1]
                    c = random.choice(valid_list)
                    add_edge(u, v, c)
        
        return cur_edges

    # Main loop
    iterations = 0
    orders = ['bipartite', 'random', 'random']
    
    while time_ok():
        iterations += 1
        seed = iterations
        order = orders[iterations % len(orders)]
        adj, color_mat, neighbors, edges = run_greedy_init(seed, order=order, prefer_formula=True)
        
        # Local search
        improved_edges = local_search(adj, color_mat, neighbors, edges, seed)
        
        # Final polish
        polished_edges = final_polish(adj, color_mat, neighbors, improved_edges)
        
        if len(polished_edges) > best_count:
            best_count = len(polished_edges)
            best_edges = [e[:] for e in polished_edges]
        
        # If we have a very good result, do a few more focused iterations
        if best_count >= 150 and iterations % 5 == 0:
            # Try to improve from the best result
            adj2 = [[False] * n for _ in range(n)]
            color_mat2 = [[-1] * n for _ in range(n)]
            neighbors2 = [set() for _ in range(n)]
            for u, v, c in best_edges:
                adj2[u][v] = True
                adj2[v][u] = True
                color_mat2[u][v] = c
                color_mat2[v][u] = c
                neighbors2[u].add(v)
                neighbors2[v].add(u)
            
            improved2 = local_search(adj2, color_mat2, neighbors2, best_edges, seed + 1000)
            polished2 = final_polish(adj2, color_mat2, neighbors2, improved2)
            
            if len(polished2) > best_count:
                best_count = len(polished2)
                best_edges = [e[:] for e in polished2]
    
    return best_edges
