"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random
    start_time = time.time()
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def color_graph(edges, deadline):
        if not edges:
            return []
        
        m = len(edges)
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Precompute covers: for edge (u,v), cover = {u,v} ∪ N(u) ∪ N(v)
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)
        
        # Build conflict bitmasks
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        
        # Conflict degree
        conflict_deg = [bin(cm).count('1') for cm in conflict_masks]
        
        # --- Quick greedy coloring for initial solution ---
        def greedy_color(order):
            color = [-1] * m
            color_used_by = []
            max_color = 0
            for i in order:
                cm = conflict_masks[i]
                taken = set()
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            return color, max_color
        
        # Generate a few orderings for initial solution
        orderings = []
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i], reverse=True))
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i]))
        rng_init = Random(42)
        for _ in range(10):
            order = list(range(m))
            rng_init.shuffle(order)
            orderings.append(order)
        
        best_color = None
        best_max_color = m + 1
        for order in orderings:
            color, max_color = greedy_color(order)
            if max_color < best_max_color:
                best_max_color = max_color
                best_color = color[:]
        
        # --- Local search: hill-climbing with restarts to reduce colors ---
        # State: color assignment. Objective: minimize max color used.
        # Moves: 
        #   1. Recolor an edge to a different (lower) color if feasible
        #   2. Try to merge two color classes
        #   3. Move an edge from one class to another to enable a merge
        
        def can_recolor(color_arr, edge_idx, new_color, color_members):
            """Check if edge_idx can be recolored to new_color."""
            cm = conflict_masks[edge_idx]
            for j in color_members[new_color]:
                if (cm >> j) & 1:
                    return False
            return True
        
        def get_color_members(color_arr):
            """Return list of lists: color_members[c] = set of edges with color c."""
            max_c = max(color_arr) if color_arr else 0
            members = [set() for _ in range(max_c + 1)]
            for i, c in enumerate(color_arr):
                members[c].add(i)
            return members
        
        def local_search(color_arr, deadline, rng):
            """Hill-climbing local search to minimize number of colors."""
            color_arr = color_arr[:]
            max_c = max(color_arr)
            members = get_color_members(color_arr)
            
            no_improve_count = 0
            max_no_improve = 200
            
            while time.time() < deadline and no_improve_count < max_no_improve:
                improved = False
                
                # Try to eliminate the highest color class
                if max_c > 0:
                    top_color = max_c
                    top_edges = list(members[top_color])
                    
                    # Try to move each edge in top_color to a lower color
                    for i in top_edges:
                        if time.time() > deadline:
                            break
                        cm = conflict_masks[i]
                        # Try colors from 0 up to max_c-1
                        for new_c in range(max_c):
                            if new_c >= len(members):
                                break
                            if can_recolor(color_arr, i, new_c, members):
                                # Move edge i to new_c
                                members[top_color].discard(i)
                                members[new_c].add(i)
                                color_arr[i] = new_c
                                improved = True
                                break
                        if improved:
                            break
                    
                    # If top color is now empty, reduce max_c
                    if not members[top_color]:
                        max_c -= 1
                        improved = True
                    elif improved:
                        # Rebuild members if needed (should be fine)
                        pass
                
                # Try to merge any two color classes
                if not improved and max_c > 0:
                    colors_used = [c for c in range(max_c + 1) if members[c]]
                    for idx1 in range(len(colors_used) - 1):
                        if time.time() > deadline:
                            break
                        c1 = colors_used[idx1]
                        for c2 in colors_used[idx1 + 1:]:
                            # Check if all edges in c1 are compatible with all in c2
                            can_merge = True
                            for i in members[c1]:
                                cm = conflict_masks[i]
                                for j in members[c2]:
                                    if (cm >> j) & 1:
                                        can_merge = False
                                        break
                                if not can_merge:
                                    break
                            if can_merge:
                                # Merge c2 into c1
                                for i in members[c2]:
                                    color_arr[i] = c1
                                    members[c1].add(i)
                                members[c2].clear()
                                improved = True
                                break
                        if improved:
                            break
                    
                    # Renumber colors to be contiguous
                    if improved:
                        color_map = {}
                        next_c = 0
                        for c in range(max_c + 1):
                            if members[c]:
                                color_map[c] = next_c
                                next_c += 1
                        for i in range(m):
                            color_arr[i] = color_map[color_arr[i]]
                        max_c = next_c - 1
                        members = get_color_members(color_arr)
                
                if not improved:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
            
            return color_arr, max(color_arr) if m > 0 else 0
        
        # Run local search with multiple random restarts
        best_result = (best_color[:], best_max_color)
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            rng_ls = Random(restart * 137 + 7)
            
            # Start from the best greedy solution
            start_color = best_result[0][:]
            
            # Optionally perturb
            if restart > 0 and m > 0:
                # Randomly reassign a few edges
                num_perturb = min(5, m // 4)
                for _ in range(num_perturb):
                    i = rng_ls.randrange(m)
                    # Find a valid color for i
                    cm = conflict_masks[i]
                    old_c = start_color[i]
                    # Try to move to a random other color
                    for _ in range(10):
                        new_c = rng_ls.randrange(best_result[1] + 1)
                        if new_c == old_c:
                            continue
                        # Check feasibility
                        feasible = True
                        for j in range(m):
                            if start_color[j] == new_c and (cm >> j) & 1:
                                feasible = False
                                break
                        if feasible:
                            start_color[i] = new_c
                            break
            
            result_color, result_max = local_search(start_color, deadline, rng_ls)
            
            if result_max < best_result[1]:
                best_result = (result_color[:], result_max)
                # Update best_color for future restarts
                best_color = result_color[:]
                best_max_color = result_max
        
        return best_result[0]
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        elapsed = time.time() - start_time
        remaining = 145 - elapsed
        if remaining < 5:
            # Fallback to simple greedy
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
        
        deadline = start_time + 145 - 5  # 5s safety margin
        edges = instance(*spec)
        colors = color_graph(edges, deadline)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
