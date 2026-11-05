import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    
    # Precompute for each edge (u,v), the list of (a, b, ca_edge_idx, cb_edge_idx) 
    # where a is neighbor of u, b is neighbor of v, a != b, a != v, b != u, and edge (a,b) exists
    # This is used to check 4-cycles containing (u,v): the cycle is a-u-v-b-a
    # Colors: color(u,a), color(u,v), color(v,b), color(a,b) must all be distinct
    
    # For efficiency, precompute for each pair (u,v) the list of "conflict quadruples"
    # A quadruple is (a, b) where a~u, b~v, a!=b, a!=v, b!=u, and a~b
    # When adding edge (u,v) with color c, we need for all such (a,b):
    #   color(u,a) != c, color(v,b) != c, color(a,b) != c, color(u,a) != color(v,b), color(u,a) != color(a,b), color(v,b) != color(a,b)
    # The last three are independent of c, so they must already be satisfied.
    # So forbidden colors for (u,v) = {color(u,a), color(v,b), color(a,b) for all valid (a,b)}
    
    def check_validity(edges):
        """Verify all 4-cycles are rainbow."""
        adj = [[-1] * n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        
        for u in range(n):
            for v in range(n):
                if u == v or adj[u][v] == -1:
                    continue
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True
    
    def greedy_build(seed, init_edges=None):
        """Build a valid edge-colored graph greedily."""
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
            existing = set((min(u, v), max(u, v)) for u, v, c in edges)
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
                        # Check if existing colors already conflict (independent of new edge color)
                        if ca == cb or cb == cv or ca == cv:
                            feasible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not feasible:
                    break
            
            if feasible:
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
    
    best_count = 0
    best_edges = []
    
    # Phase 1: Many greedy restarts with different seeds
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search - remove k edges, refill
    current_best = best_edges[:]
    
    # Also try with different initial structures
    for restart_round in range(50):
        if time.time() > deadline - 20:
            break
        
        # Try removing random edges and refilling
        if len(current_best) > 5:
            k = random.randint(2, 10)
            to_remove = random.sample(current_best, min(k, len(current_best)))
            rem_set = set((min(u, v), max(u, v)) for u, v, c in to_remove)
            base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
            
            seed = random.randint(0, 10**9)
            new_edges = greedy_build(seed, base)
            
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges[:]
                current_best = new_edges[:]
    
    # Phase 3: Re-coloring search
    if time.time() < deadline - 10:
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
                    for new_c in range(num_colors):
                        if new_c == old_c:
                            continue
                        # Check if re-coloring is valid
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
                                    if ca == new_c or new_c == cv or cv == cb or ca == cb:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Try to add one missing edge
                        added = False
                        for (wu, wv) in all_edges:
                            if adj[wu][wv] != -1:
                                continue
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
                            candidates = [c for c in range(num_colors) if c not in forbidden_add]
                            if candidates:
                                nc = rng.choice(candidates)
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
                    if added:
                        break
                
                # After re-coloring, try to add more edges greedily
                if improved:
                    existing = set((min(u, v), max(u, v)) for u, v, c in edges)
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
        
        recolored = recolor_search(current_best[:], 42, deadline - 5)
        if len(recolored) > best_count:
            best_count = len(recolored)
            best_edges = recolored[:]
            current_best = recolored[:]
    
    # Final verification
    if best_edges and not check_validity(best_edges):
        # Find a valid fallback
        best_edges = []
        for i in range(100):
            candidate = greedy_build(1000 + i)
            if check_validity(candidate):
                best_edges = candidate
                break
        if not best_edges:
            best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
