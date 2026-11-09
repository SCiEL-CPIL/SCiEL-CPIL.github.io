import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 145
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle
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
            witness[(u, v)] = np.array(wl, dtype=np.int32)
            witness[(v, u)] = witness[(u, v)]
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def get_forbidden_fast(u, v, adj):
        """Vectorized forbidden color computation."""
        wl = witness[(u, v)]
        if len(wl) == 0:
            return set()
        
        xs = wl[:, 0]
        ys = wl[:, 1]
        
        # Check which 4-cycles are complete
        valid = (adj[v, xs] != -1) & (adj[xs, ys] != -1) & (adj[ys, u] != -1)
        
        if not np.any(valid):
            return set()
        
        c1 = adj[v, xs[valid]]
        c2 = adj[xs[valid], ys[valid]]
        c3 = adj[ys[valid], u]
        
        # Check for duplicate colors in existing edges (shouldn't happen in valid state)
        if np.any(c1 == c2) or np.any(c1 == c3) or np.any(c2 == c3):
            return set(range(n))
        
        return set(np.concatenate([c1, c2, c3]))
    
    def is_recolor_valid_fast(u, v, new_c, adj):
        """Vectorized recolor validity check."""
        wl = witness[(u, v)]
        if len(wl) == 0:
            return True
        
        xs = wl[:, 0]
        ys = wl[:, 1]
        
        valid = (adj[v, xs] != -1) & (adj[xs, ys] != -1) & (adj[ys, u] != -1)
        
        if not np.any(valid):
            return True
        
        c1 = adj[v, xs[valid]]
        c2 = adj[xs[valid], ys[valid]]
        c3 = adj[ys[valid], u]
        
        return not np.any(c1 == new_c) and not np.any(c2 == new_c) and not np.any(c3 == new_c)
    
    def compute_energy(adj):
        """Compute energy: number of edges missing + penalty for blocked edges."""
        cnt = count_edges(adj)
        missing = n * (n - 1) // 2 - cnt
        return missing
    
    def try_add_edge(u, v, adj, strategy='random'):
        """Try to add edge (u,v) with given color strategy."""
        if adj[u, v] != -1:
            return False
        
        forb = get_forbidden_fast(u, v, adj)
        if len(forb) >= n:
            return False
        
        available = [c for c in range(n) if c not in forb]
        if not available:
            return False
        
        if strategy == 'random':
            c = random.choice(available)
        elif strategy == 'least_used':
            # Choose color least used in neighborhood
            counts = np.zeros(n, dtype=int)
            for x in range(n):
                if x != u and x != v and adj[u, x] != -1:
                    counts[adj[u, x]] += 1
                if x != u and x != v and adj[v, x] != -1:
                    counts[adj[v, x]] += 1
            c = min(available, key=lambda col: counts[col])
        else:
            c = random.choice(available)
        
        adj[u, v] = c
        adj[v, u] = c
        return True
    
    def simulated_annealing(adj, max_iters=5000, t_start=1.0, t_end=0.01):
        """Simulated annealing to maximize edge count."""
        current_energy = compute_energy(adj)
        best_adj = adj.copy()
        best_energy = current_energy
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            t = t_start * (t_end / t_start) ** (it / max_iters)
            
            # Choose action: add edge, remove edge, or recolor
            action = random.random()
            
            if action < 0.4:
                # Try to add a missing edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
                if not missing:
                    continue
                u, v = random.choice(missing)
                forb = get_forbidden_fast(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    c = random.choice(available)
                    adj[u, v] = c
                    adj[v, u] = c
                    new_energy = compute_energy(adj)
                    delta = new_energy - current_energy
                    if delta <= 0 or random.random() < np.exp(-delta / t):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_adj = adj.copy()
                    else:
                        adj[u, v] = -1
                        adj[v, u] = -1
                        
            elif action < 0.6:
                # Recolor an existing edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
                if not existing:
                    continue
                u, v = random.choice(existing)
                old_c = adj[u, v]
                new_c = random.randint(0, n - 1)
                if new_c == old_c:
                    continue
                if is_recolor_valid_fast(u, v, new_c, adj):
                    adj[u, v] = new_c
                    adj[v, u] = new_c
                    # Energy unchanged for recoloring, but may enable future additions
                    # Keep it (always accept valid recolors)
                else:
                    # Reject invalid recolor
                    pass
                    
            else:
                # Coordinated recoloring: recolor 2-3 edges together
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] != -1]
                if len(existing) < 3:
                    continue
                
                k = random.randint(2, 3)
                edges_to_recolor = random.sample(existing, k)
                
                # Try all combinations of new colors (sampled)
                found = False
                for _ in range(10):
                    new_colors = [random.randint(0, n - 1) for _ in range(k)]
                    
                    # Check if all recolorings are valid simultaneously
                    # First apply tentatively
                    old_colors = []
                    valid_all = True
                    for (u, v), nc in zip(edges_to_recolor, new_colors):
                        if nc == adj[u, v]:
                            continue
                        old_colors.append((u, v, adj[u, v]))
                        # Check validity considering other recolors
                        # Simplified: check individually, then verify combined
                        if not is_recolor_valid_fast(u, v, nc, adj):
                            valid_all = False
                            break
                        adj[u, v] = nc
                        adj[v, u] = nc
                    
                    if valid_all:
                        # Verify combined validity
                        combined_valid = True
                        for (u, v) in edges_to_recolor:
                            wl = witness[(u, v)]
                            if len(wl) == 0:
                                continue
                            xs = wl[:, 0]
                            ys = wl[:, 1]
                            valid_mask = (adj[v, xs] != -1) & (adj[xs, ys] != -1) & (adj[ys, u] != -1)
                            if np.any(valid_mask):
                                c1 = adj[v, xs[valid_mask]]
                                c2 = adj[xs[valid_mask], ys[valid_mask]]
                                c3 = adj[ys[valid_mask], u]
                                cu = adj[u, v]
                                if np.any(c1 == cu) or np.any(c2 == cu) or np.any(c3 == cu):
                                    combined_valid = False
                                    break
                                if np.any(c1 == c2) or np.any(c1 == c3) or np.any(c2 == c3):
                                    combined_valid = False
                                    break
                        
                        if combined_valid:
                            found = True
                            break
                        else:
                            # Undo
                            for (u, v, oc) in old_colors:
                                adj[u, v] = oc
                                adj[v, u] = oc
                    
                    if not valid_all:
                        for (u, v, oc) in old_colors:
                            adj[u, v] = oc
                            adj[v, u] = oc
                
                if found:
                    # Try to add an edge now
                    missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
                    if missing:
                        random.shuffle(missing)
                        for u, v in missing[:5]:
                            if try_add_edge(u, v, adj, 'random'):
                                new_energy = compute_energy(adj)
                                if new_energy < best_energy:
                                    best_energy = new_energy
                                    best_adj = adj.copy()
                                break
        
        return best_adj
    
    best_count = 0
    best_adj = None
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with random spanning tree
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i - 1)]
            c = random.randint(0, n - 1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy edge addition
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            try_add_edge(u, v, adj, 'least_used')
        
        # Simulated annealing
        adj = simulated_annealing(adj, max_iters=3000, t_start=1.0, t_end=0.05)
        
        # Final greedy pass
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            try_add_edge(u, v, adj, 'least_used')
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
        
        if best_count > 175:
            break
    
    # Final polish
    if best_adj is not None and time.time() < deadline - 10:
        adj = best_adj.copy()
        adj = simulated_annealing(adj, max_iters=5000, t_start=0.5, t_end=0.01)
        
        for u, v in all_edges:
            if time.time() > deadline - 2:
                break
            if adj[u, v] != -1:
                continue
            try_add_edge(u, v, adj, 'least_used')
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Convert to result
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
