import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_forbidden_colors(u, v, adj, color_mat, neighbors):
        """
        Returns the set of colors forbidden for edge (u, v).
        """
        forbidden = set()
        for w in neighbors[v]:
            if w == u: continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # If the existing 3 edges have duplicate colors, the current graph is invalid.
                # But we assume the graph is always valid during the search.
                # So we just add these colors to the forbidden set.
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
        return forbidden

    def is_valid_color_change(u, v, new_c, adj, color_mat, neighbors):
        """
        Check if changing color of edge (u, v) to new_c keeps the graph valid.
        """
        old_c = color_mat[u][v]
        if old_c == new_c:
            return True
            
        # Temporarily set new color
        color_mat[u][v] = new_c
        color_mat[v][u] = new_c
        
        valid = True
        
        # Check all 4-cycles involving edge (u, v)
        # Cycle: u-v-w-x-u
        for w in neighbors[v]:
            if w == u: continue
            c_vw = color_mat[v][w]
            for x in neighbors[u]:
                if x == v: continue
                if x == w: continue
                if not adj[w][x]: continue
                
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # Colors in cycle: new_c, c_vw, c_wx, c_xu
                # They must be distinct
                if len({new_c, c_vw, c_wx, c_xu}) != 4:
                    valid = False
                    break
            if not valid:
                break
        
        # Restore old color if invalid
        if not valid:
            color_mat[u][v] = old_c
            color_mat[v][u] = old_c
            
        return valid

    def run_greedy(seed):
        random.seed(seed)
        adj = [[0] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        shuffled_edges = all_possible_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
            valid_colors = [c for c in range(num_colors) if c not in forbidden]
            
            if valid_colors:
                c = random.choice(valid_colors)
                adj[u][v] = 1
                adj[v][u] = 1
                color_mat[u][v] = c
                color_mat[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
                
        return edges, adj, color_mat, neighbors

    def local_search_recolor(edges, adj, color_mat, neighbors, seed):
        """
        Local search that tries to add missing edges by recoloring existing edges.
        """
        random.seed(seed)
        edges = [e[:] for e in edges]
        adj = [[0] * n for _ in range(n)]
        color_mat = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            
        current_count = len(edges)
        missing_edges = [(u, v) for u, v in all_possible_edges if not adj[u][v]]
        
        # Time limit for local search per restart
        ls_deadline = time.time() + 5.0 
        
        while time.time() < ls_deadline and missing_edges:
            random.shuffle(missing_edges)
            progress = False
            
            for u, v in missing_edges:
                forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
                valid_colors = [c for c in range(num_colors) if c not in forbidden]
                
                if valid_colors:
                    # Can add directly
                    c = random.choice(valid_colors)
                    adj[u][v] = 1
                    adj[v][u] = 1
                    color_mat[u][v] = c
                    color_mat[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    missing_edges.remove((u, v))
                    current_count += 1
                    progress = True
                    break # Restart loop to re-evaluate others
            
            if progress:
                continue
                
            # If no edge could be added directly, try recoloring to enable one
            # Pick a random missing edge
            if not missing_edges:
                break
                
            u, v = random.choice(missing_edges)
            forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
            
            # Find a 4-cycle causing the blockage
            # We need to find a cycle u-w-x-v where colors forbid the needed space
            # Actually, if all 20 colors are forbidden, it's saturated.
            # We pick a random edge involved in a 4-cycle with (u,v) and try to recolor it.
            
            # Identify edges in 4-cycles with u-v
            candidate_edges = set()
            for w in neighbors[v]:
                if w == u: continue
                for x in neighbors[u]:
                    if x == v: continue
                    if x == w: continue
                    if not adj[w][x]: continue
                    # Cycle u-v-w-x-u
                    candidate_edges.add((v, w))
                    candidate_edges.add((w, x))
                    candidate_edges.add((x, u))
            
            if not candidate_edges:
                continue
                
            candidates = list(candidate_edges)
            random.shuffle(candidates)
            
            recolor_success = False
            for cu, cv in candidates:
                # Ensure consistent ordering for color_mat access
                if cu > cv:
                    cu, cv = cv, cu
                    
                old_c = color_mat[cu][cv]
                
                # Try a random new color
                new_c = random.choice([c for c in range(num_colors) if c != old_c])
                
                if is_valid_color_change(cu, cv, new_c, adj, color_mat, neighbors):
                    # Accept recoloring
                    # Now try to add (u, v) again
                    forbidden_new = get_forbidden_colors(u, v, adj, color_mat, neighbors)
                    valid_colors_new = [c for c in range(num_colors) if c not in forbidden_new]
                    
                    if valid_colors_new:
                        c_add = random.choice(valid_colors_new)
                        adj[u][v] = 1
                        adj[v][u] = 1
                        color_mat[u][v] = c_add
                        color_mat[v][u] = c_add
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c_add])
                        missing_edges.remove((u, v))
                        current_count += 1
                        recolor_success = True
                        break
                    else:
                        # Recoloring was valid but didn't help add (u,v)?
                        # This can happen if other cycles still forbid all colors.
                        # We keep the recoloring as it might help other edges.
                        # But we break to restart the main loop.
                        recolor_success = True
                        break
                else:
                    # is_valid_color_change restores color if invalid, so no action needed
                    pass
            
            if recolor_success:
                # Restart the while loop to re-evaluate all missing edges
                continue
                
            # If we tried everything and couldn't add or recolor usefully, break
            break
            
        return edges, adj, color_mat, neighbors

    restarts = 0
    # Run as many restarts as possible
    while time.time() < deadline:
        edges, adj, color_mat, neighbors = run_greedy(seed=restarts)
        current_count = len(edges)
        
        # Try local search to improve
        if time.time() < deadline - 10:
            improved_edges, _, _, _ = local_search_recolor(edges, adj, color_mat, neighbors, seed=restarts)
            if len(improved_edges) > current_count:
                edges = improved_edges
                current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = edges
            
        if best_count == 190:
            break
            
        restarts += 1

    return best_edges
