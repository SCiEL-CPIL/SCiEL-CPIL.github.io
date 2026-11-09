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
    
    # Precompute witnesses: for edge (u,v), list of (x,y) such that v-x-y-u is a 4-cycle
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
        """Return (forbidden_set, blocked_flag) for adding edge (u,v)."""
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

    def find_valid_color(adj, u, v, deterministic=False):
        """Find a valid color for edge (u,v)."""
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
        if not available:
            return -1
        if deterministic:
            color_counts = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] != -1:
                        color_counts[adj[i, j]] += 1
            best_c = min(available, key=lambda c: color_counts[c])
            return best_c
        return random.choice(available)

    def can_recolor_edge_fast(adj, u, v, new_c):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles valid."""
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

    def local_search(adj, max_rounds=50):
        """Iteratively try to add missing edges via recoloring. Returns edge count."""
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
                # Try recoloring a neighbor edge
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
        
        return np.sum(adj != -1) // 2

    def verify_adj(adj):
        """Verify that all 4-cycles in adj are rainbow."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == -1:
                    continue
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v, x] == -1:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if adj[x, y] == -1:
                            continue
                        if adj[y, u] == -1:
                            continue
                        c1 = adj[u, v]
                        c2 = adj[v, x]
                        c3 = adj[x, y]
                        c4 = adj[y, u]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True

    def csp_solve(adj, max_depth=30, max_nodes=50000):
        """CSP solver with dynamic variable ordering and forward checking."""
        # Count current edges
        current_edges = np.sum(adj != -1) // 2
        
        # Get all missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        
        if not missing:
            return current_edges
        
        # State: adj copy, list of unassigned edges, assignments
        adj_copy = adj.copy()
        unassigned = list(missing)
        
        # Forward checking: for each unassigned edge, compute which colors are still possible
        def compute_domains(adj_state, unassigned_list):
            domains = {}
            for (u, v) in unassigned_list:
                forbidden, blocked = get_forbidden_and_blocked(adj_state, u, v)
                if blocked:
                    domains[(u, v)] = set()
                else:
                    domains[(u, v)] = set(range(n)) - forbidden
            return domains
        
        def select_variable(unassigned_list, domains):
            """MRV heuristic: pick variable with fewest remaining values."""
            best_var = None
            best_count = float('inf')
            for var in unassigned_list:
                if var not in domains:
                    continue
                count = len(domains[var])
                if count == 0:
                    return var  # Failed
                if count < best_count:
                    best_count = count
                    best_var = var
            return best_var
        
        def forward_check(adj_state, unassigned_list, domains, assigned_var, color):
            """Update domains after assigning color to assigned_var."""
            new_domains = {k: set(v) for k, v in domains.items()}
            u, v = assigned_var
            adj_state[u, v] = color
            adj_state[v, u] = color
            
            for (a, b) in unassigned_list:
                if (a, b) == assigned_var:
                    continue
                old_domain = new_domains.get((a, b), set())
                if not old_domain:
                    continue
                
                forbidden, blocked = get_forbidden_and_blocked(adj_state, a, b)
                if blocked:
                    new_domains[(a, b)] = set()
                else:
                    new_domains[(a, b)] = old_domain - forbidden
            
            # Remove assigned_var from unassigned
            new_unassigned = [var for var in unassigned_list if var != assigned_var]
            
            return new_domains, new_unassigned, adj_state
        
        def backtrack(adj_state, unassigned_list, domains, depth, nodes_visited):
            if nodes_visited[0] > max_nodes:
                return False
            
            if not unassigned_list:
                return True
            
            nodes_visited[0] += 1
            
            var = select_variable(unassigned_list, domains)
            if var is None:
                return False
            
            if var not in domains or len(domains[var]) == 0:
                return False
            
            # Try colors in order of least constraining value
            colors = list(domains[var])
            random.shuffle(colors)
            
            for color in colors:
                new_domains, new_unassigned, new_adj = forward_check(
                    adj_state.copy(), unassigned_list, domains, var, color
                )
                
                # Check if any domain is empty (failure)
                if any(len(new_domains.get(v, set())) == 0 for v in new_unassigned):
                    continue
                
                if backtrack(new_adj, new_unassigned, new_domains, depth + 1, nodes_visited):
                    # Copy back the assignment
                    adj_state[var[0], var[1]] = color
                    adj_state[var[1], var[0]] = color
                    return True
            
            return False
        
        nodes_visited = [0]
        initial_domains = compute_domains(adj_copy, unassigned)
        
        # Remove edges with empty domains from consideration (they can't be added)
        addable = [(u, v) for (u, v) in unassigned if len(initial_domains.get((u, v), set())) > 0]
        
        if not addable:
            return current_edges
        
        # Use a subset for CSP to keep it tractable
        random.shuffle(addable)
        subset = addable[:min(20, len(addable))]
        
        # Build domains only for the subset
        subset_domains = {}
        for (u, v) in subset:
            forbidden, blocked = get_forbidden_and_blocked(adj_copy, u, v)
            if blocked:
                subset_domains[(u, v)] = set()
            else:
                subset_domains[(u, v)] = set(range(n)) - forbidden
        
        # Filter out empty domains
        active = [(u, v) for (u, v) in subset if len(subset_domains.get((u, v), set())) > 0]
        
        if not active:
            return current_edges
        
        success = backtrack(adj_copy, active, subset_domains, 0, nodes_visited)
        
        if success:
            final_edges = np.sum(adj_copy != -1) // 2
            if verify_adj(adj_copy):
                return final_edges
        
        return current_edges

    for restart in range(2000):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Build initial spanning tree with random colors
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy add edges in random order
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
        
        # Local search to add more edges
        cnt = local_search(adj)
        
        # CSP refinement to push further
        cnt_csp = csp_solve(adj)
        if cnt_csp > cnt:
            # Verify and update best if valid
            if verify_adj(adj):
                cnt = cnt_csp
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 175:
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
