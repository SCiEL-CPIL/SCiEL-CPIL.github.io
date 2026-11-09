import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n = 16
    
    # --- Helper functions ---
    
    def make_adj():
        return [set() for _ in range(n)]
    
    def add_edge(adj, edge_set, u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        # Triangle-free check
        if adj[u] & adj[v]:
            return False
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    def remove_edge(adj, edge_set, u, v):
        if u > v:
            u, v = v, u
        edge_set.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
    
    def is_3_colorable(adj):
        """Fast backtracking 3-colorability with MRV and forward checking."""
        color = [-1] * n
        domains = [set([0, 1, 2]) for _ in range(n)]
        uncolored = list(range(n))
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV: pick uncolored vertex with fewest available colors
            best_v = -1
            best_count = float('inf')
            best_idx = -1
            for i, v in enumerate(uncolored):
                cnt = len(domains[v])
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    best_idx = i
                    if cnt == 1:
                        break
            
            if best_v == -1:
                return True
            
            v = best_v
            uncolored.pop(best_idx)
            saved_domain = domains[v].copy()
            
            for c in list(domains[v]):
                color[v] = c
                consistent = True
                saved = []
                for nb in adj[v]:
                    if color[nb] == -1 and c in domains[nb]:
                        saved.append((nb, c))
                        domains[nb].discard(c)
                        if len(domains[nb]) == 0:
                            consistent = False
                            break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = -1
                for nb, c in saved:
                    domains[nb].add(c)
            
            uncolored.insert(best_idx, v)
            domains[v] = saved_domain
            return False
        
        return backtrack()
    
    def find_large_independent_set(adj, vertices):
        """Greedy: repeatedly pick vertex with fewest neighbors in remaining set."""
        remaining = set(vertices)
        independent = []
        while remaining:
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            independent.append(best_v)
            remaining.remove(best_v)
            for nb in adj[best_v]:
                if nb in remaining:
                    remaining.discard(nb)
        return independent
    
    def greedy_add_edges(adj, edge_set, time_limit):
        """Repeatedly add the cheapest valid edge (lowest degree sum) until no more can be added or time runs out."""
        while time.time() < time_limit:
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in edge_set:
                        continue
                    if adj[u] & adj[v]:
                        continue
                    cost = len(adj[u]) + len(adj[v])
                    candidates.append((cost, u, v))
            
            if not candidates:
                break
            
            candidates.sort()
            added = False
            for cost, u, v in candidates:
                if add_edge(adj, edge_set, u, v):
                    added = True
                    break
            
            if not added:
                break
    
    def local_search_swaps(adj, edge_set, time_limit):
        """Try edge swaps: remove one edge, add another that was previously blocked."""
        best_count = len(edge_set)
        
        while time.time() < time_limit:
            # Collect all edges
            edges = list(edge_set)
            random.shuffle(edges)
            
            improved = False
            for e1 in edges:
                if time.time() > time_limit:
                    break
                u1, v1 = e1
                
                # Find edges blocked by e1
                potential_new = []
                for x in adj[v1]:
                    if (u1, x) not in edge_set and x != u1:
                        if not (adj[u1] & adj[x]):
                            potential_new.append((u1, x))
                for y in adj[u1]:
                    if (v1, y) not in edge_set and y != v1:
                        if not (adj[v1] & adj[y]):
                            potential_new.append((v1, y))
                
                random.shuffle(potential_new)
                
                for u2, v2 in potential_new:
                    if time.time() > time_limit:
                        break
                    # Remove e1
                    remove_edge(adj, edge_set, u1, v1)
                    
                    # Try to add e2
                    if add_edge(adj, edge_set, u2, v2):
                        if not is_3_colorable(adj):
                            # Check if we can add more edges after this swap
                            greedy_add_edges(adj, edge_set, min(time.time() + 0.5, time_limit))
                            if len(edge_set) > best_count:
                                best_count = len(edge_set)
                                improved = True
                        else:
                            # Undo: remove e2, re-add e1
                            remove_edge(adj, edge_set, u2, v2)
                            add_edge(adj, edge_set, u1, v1)
                    else:
                        # Re-add e1
                        add_edge(adj, edge_set, u1, v1)
                
                if improved:
                    break
            
            if not improved:
                # Try random restarts with perturbation
                if time.time() > time_limit - 5:
                    break
                
                # Perturb: remove a random edge and re-run greedy
                if edges:
                    e_rand = random.choice(edges)
                    remove_edge(adj, edge_set, e_rand[0], e_rand[1])
                    greedy_add_edges(adj, edge_set, min(time.time() + 1.0, time_limit))
                    if len(edge_set) > best_count:
                        best_count = len(edge_set)
    
    # --- Main construction ---
    
    # Start with Grötzsch graph (Mycielski of C5) on vertices 0..10
    edge_set = set()
    adj = make_adj()
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(adj, edge_set, i, (i + 1) % 5)
    
    # Mycielski: vertices 5..9
    for i in range(5):
        add_edge(adj, edge_set, i, 5 + (i + 1) % 5)
        add_edge(adj, edge_set, i, 5 + (i - 1) % 5)
    
    # Vertex 10 connected to 5..9
    for i in range(5):
        add_edge(adj, edge_set, 10, 5 + i)
    
    # Add vertices 11..15 by connecting to large independent sets
    for v in range(11, 16):
        if time.time() > deadline - 30:
            break
        indep = find_large_independent_set(adj, range(v))
        for u in indep:
            add_edge(adj, edge_set, v, u)
    
    # Greedy edge addition
    greedy_add_edges(adj, edge_set, deadline - 30)
    
    # Local search with swaps
    local_search_swaps(adj, edge_set, deadline - 5)
    
    # Final attempt: try all possible single-edge additions one more time
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline - 2:
                break
            if (u, v) in edge_set:
                continue
            if adj[u] & adj[v]:
                continue
            if add_edge(adj, edge_set, u, v):
                if is_3_colorable(adj):
                    remove_edge(adj, edge_set, u, v)
                else:
                    # Keep it, try to add more
                    greedy_add_edges(adj, edge_set, deadline - 2)
    
    # Convert to list of edges
    result = [[u, v] for (u, v) in edge_set]
    return result
