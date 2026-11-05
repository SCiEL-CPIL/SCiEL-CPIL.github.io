import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    # Precompute all 4-cycles (u, v, w, x) where edges (u,v), (v,w), (w,x), (x,u) exist
    # To check if a 4-cycle is rainbow: colors of these 4 edges must be distinct
    
    # We'll build the graph greedily with local search improvement
    
    best_count = 0
    best_edges = []
    
    def greedy_build(seed, init_edges=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
        
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            forbidden = set()
            feasible = True
            
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        if ca == cb or cb == cv or ca == cv:
                            feasible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not feasible:
                    break
            
            if feasible:
                # Pick a color not in forbidden; prefer least-used color for balance
                if forbidden:
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges
    
    def check_4cycle_rainbow(adj, u, v, w, x):
        """Check if edges (u,v),(v,w),(w,x),(x,u) all exist and form a rainbow 4-cycle."""
        c1 = adj[u][v]
        c2 = adj[v][w]
        c3 = adj[w][x]
        c4 = adj[x][u]
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            return False
        return len({c1, c2, c3, c4}) == 4
    
    def recolor_search(edges, seed, time_limit):
        """Try to improve by re-coloring existing edges to unlock new additions."""
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try re-coloring each edge
            for idx, (u, v, old_c) in enumerate(edges):
                if time.time() > time_limit:
                    break
                # Try each new color for this edge
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    # Check if re-coloring is valid: all 4-cycles containing (u,v) must remain rainbow
                    valid = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                # The 4-cycle is (a,u,v,b) with colors ca, new_c, cv, cb
                                if ca == new_c or new_c == cv or cv == cb or ca == cb:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Re-color is valid; check if it allows adding a new edge
                    # Try to add one missing edge
                    added = False
                    for (wu, wv) in all_edges:
                        if adj[wu][wv] != -1:
                            continue
                        # Check if we can add edge (wu, wv) with some color
                        forbidden_add = set()
                        feasible_add = True
                        for a in neighbors[wu]:
                            if a == wv:
                                continue
                            ca = adj[wu][a]
                            for b in neighbors[wv]:
                                if b == wu or b == a:
                                    continue
                                if adj[a][b] != -1:
                                    cb = adj[a][b]
                                    cv = adj[b][wv]
                                    if ca == cb or cb == cv or ca == cv:
                                        feasible_add = False
                                        break
                                    forbidden_add.add(ca)
                                    forbidden_add.add(cb)
                                    forbidden_add.add(cv)
                            if not feasible_add:
                                break
                        if not feasible_add:
                            continue
                        # Pick a color for the new edge
                        candidates = [c for c in range(num_colors) if c not in forbidden_add]
                        if candidates:
                            nc = rng.choice(candidates)
                            # Apply re-color and add
                            adj[u][v] = new_c
                            adj[v][u] = new_c
                            adj[wu][wv] = nc
                            adj[wv][wu] = nc
                            neighbors[wu].add(wv)
                            neighbors[wv].add(wu)
                            edges[idx] = [u, v, new_c]
                            edges.append([wu, wv, nc])
                            improved = True
                            added = True
                            break
                        else:
                            feasible_add = False
                        if not feasible_add:
                            continue
                    if added:
                        break
            # After re-coloring, try to add more edges greedily
            if improved:
                existing = set((min(u,v), max(u,v)) for u, v, c in edges)
                remaining = [e for e in all_edges if e not in existing]
                rng.shuffle(remaining)
                for (u, v) in remaining:
                    if time.time() > time_limit:
                        break
                    forbidden = set()
                    feasible = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                if ca == cb or cb == cv or ca == cv:
                                    feasible = False
                                    break
                                forbidden.add(ca)
                                forbidden.add(cb)
                                forbidden.add(cv)
                            if not feasible:
                                break
                    if not feasible:
                        continue
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
        
        return edges
    
    # Phase 1: Greedy random restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Phase 2: Local search — remove k edges, refill greedily
    current_best = best_edges[:]
    
    while time.time() < deadline:
        k = random.randint(1, 8)
        if len(current_best) <= k:
            break
        
        to_remove = random.sample(current_best, k)
        rem_set = set((min(u,v), max(u,v)) for u, v, c in to_remove)
        base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
        
        seed = random.randint(0, 10**9)
        new_edges = greedy_build(seed, base)
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
            current_best = new_edges[:]
    
    # Phase 3: Re-coloring search to unlock new edges
    if time.time() < deadline - 5:
        recolor_limit = min(deadline - 2, time.time() + 30)
        recolored = recolor_search(current_best[:], 42, recolor_limit)
        if len(recolored) > best_count:
            best_count = len(recolored)
            best_edges = recolored
            current_best = recolored[:]
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
