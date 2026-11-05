import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin for safety
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def check_validity(edges):
        """Verify every 4-cycle is rainbow."""
        if not edges:
            return True
        adj = [[-1]*n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        # Check all 4-cycles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] == -1:
                    continue
                for w in range(v+1, n):
                    if adj[v][w] == -1:
                        continue
                    # Find x such that u-x and w-x exist
                    # and x != u, v, w
                    common = []
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[u][x] != -1 and adj[w][x] != -1:
                            common.append(x)
                    for x in common:
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    def greedy_build(seed, initial_edges=None, order_type='random', color_strategy='random'):
        """
        Builds a graph using greedy edge addition.
        """
        rng = random.Random(seed)
        
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if initial_edges:
            for e in initial_edges:
                u, v, c = e
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing_set = set((min(u,v), max(u,v)) for u,v,_ in edges)
            remaining_edges = [e for e in all_edges if e not in existing_set]
        else:
            remaining_edges = all_edges[:]
            
        if order_type == 'random':
            rng.shuffle(remaining_edges)
        elif order_type == 'natural':
            pass
        elif order_type == 'reverse':
            remaining_edges.reverse()
            
        for (u, v) in remaining_edges:
            impossible = False
            forbidden_colors = set()
            
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
                            impossible = True
                            break
                        forbidden_colors.add(ca)
                        forbidden_colors.add(cb)
                        forbidden_colors.add(cv)
                if impossible:
                    break
            
            if not impossible:
                # Smarter color selection: minimize frequency among neighbors
                if color_strategy == 'min_freq':
                    color_freq = [0] * num_colors
                    for a in neighbors[u]:
                        c = adj[u][a]
                        if c != -1:
                            color_freq[c] += 1
                    for b in neighbors[v]:
                        c = adj[v][b]
                        if c != -1:
                            color_freq[c] += 1
                    # Choose color not in forbidden with min freq, random tiebreak
                    valid_colors = [c for c in range(num_colors) if c not in forbidden_colors]
                    if valid_colors:
                        rng.shuffle(valid_colors)
                        min_f = min(color_freq[c] for c in valid_colors)
                        candidates = [c for c in valid_colors if color_freq[c] == min_f]
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    colors = list(range(num_colors))
                    rng.shuffle(colors)
                    placed = False
                    for c in colors:
                        if c not in forbidden_colors:
                            adj[u][v] = c
                            adj[v][u] = c
                            neighbors[u].add(v)
                            neighbors[v].add(u)
                            edges.append([u, v, c])
                            placed = True
                            break
        
        return edges

    def simulated_annealing(seed, initial_edges=None, max_iters=5000):
        """
        SA over edge additions/removals/recolors.
        State: list of [u,v,c]
        Moves: add edge, remove edge, recolor edge
        """
        rng = random.Random(seed)
        
        if initial_edges:
            state = [e[:] for e in initial_edges]
        else:
            # Start with a greedy build
            state = greedy_build(seed, None, 'random', 'min_freq')
        
        adj = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in state:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        best_state = state[:]
        best_len = len(state)
        
        def is_valid_state():
            """Check if current state is valid (all 4-cycles rainbow)."""
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v] == -1:
                        continue
                    for w in range(v+1, n):
                        if adj[v][w] == -1:
                            continue
                        for x in range(n):
                            if x == u or x == v or x == w:
                                continue
                            if adj[u][x] != -1 and adj[w][x] != -1:
                                c1 = adj[u][v]
                                c2 = adj[v][w]
                                c3 = adj[w][x]
                                c4 = adj[x][u]
                                if len({c1, c2, c3, c4}) != 4:
                                    return False
            return True
        
        # Verify initial state
        if not is_valid_state():
            # Fallback to greedy
            state = greedy_build(seed, None, 'random', 'min_freq')
            adj = [[-1]*n for _ in range(n)]
            neighbors = [set() for _ in range(n)]
            for u, v, c in state:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
            best_state = state[:]
            best_len = len(state)
        
        temp = 1.0
        cooling = 0.9995
        edge_set = set((min(u,v), max(u,v)) for u,v,_ in state)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            accepted = False
            
            if move_type < 0.4:
                # Try to add a random missing edge
                missing = [(u,v) for u,v in all_edges if (u,v) not in edge_set]
                if not missing:
                    continue
                u, v = rng.choice(missing)
                
                # Check if we can add with some color
                impossible = False
                forbidden = set()
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
                                impossible = True
                                break
                            forbidden.add(ca)
                            forbidden.add(cb)
                            forbidden.add(cv)
                    if impossible:
                        break
                
                if not impossible:
                    valid_colors = [c for c in range(num_colors) if c not in forbidden]
                    if valid_colors:
                        c = rng.choice(valid_colors)
                        # Apply
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edge_set.add((u,v))
                        state.append([u, v, c])
                        accepted = True
                        # Check validity
                        if is_valid_state():
                            if len(state) > best_len:
                                best_len = len(state)
                                best_state = state[:]
                            # Keep
                        else:
                            # Undo
                            adj[u][v] = -1
                            adj[v][u] = -1
                            neighbors[u].discard(v)
                            neighbors[v].discard(u)
                            edge_set.discard((u,v))
                            state.pop()
                            accepted = False
                            
            elif move_type < 0.7:
                # Try to recolor a random edge
                if not state:
                    continue
                idx = rng.randrange(len(state))
                u, v, old_c = state[idx]
                new_c = rng.randrange(num_colors)
                if new_c == old_c:
                    continue
                
                # Apply recolor
                adj[u][v] = new_c
                adj[v][u] = new_c
                state[idx][2] = new_c
                
                if is_valid_state():
                    # Keep
                    pass
                else:
                    # Undo
                    adj[u][v] = old_c
                    adj[v][u] = old_c
                    state[idx][2] = old_c
                    accepted = False
                    
            else:
                # Try to remove a random edge
                if not state:
                    continue
                idx = rng.randrange(len(state))
                u, v, c = state[idx]
                
                # Remove
                adj[u][v] = -1
                adj[v][u] = -1
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                edge_set.discard((u,v))
                state.pop(idx)
                accepted = True
                # Removal is always valid
                # We don't track best on removal since it reduces size
            
            temp *= cooling
        
        return best_state, best_len

    # Phase 1: Greedy with multiple strategies
    order_types = ['random', 'natural', 'reverse']
    color_strategies = ['random', 'min_freq']
    num_restarts = 50
    
    for ot in order_types:
        for cs in color_strategies:
            for i in range(num_restarts):
                if time.time() > deadline:
                    break
                edges = greedy_build(i * 100 + len(ot) + len(cs), None, ot, cs)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, 0] for v in range(1, 20)]
        best_count = len(best_edges)

    # Phase 2: Simulated Annealing refinement
    current_best = best_edges[:]
    max_sa_iterations = 2000
    
    for sa_iter in range(max_sa_iterations):
        if time.time() > deadline:
            break
        
        # Occasionally perturb current best
        if sa_iter % 10 == 0 and len(current_best) > 5:
            k = random.randint(1, 3)
            edges_to_remove = random.sample(current_best, min(k, len(current_best)))
            remove_set = set((min(u,v), max(u,v)) for u,v,_ in edges_to_remove)
            base_edges = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
        else:
            base_edges = current_best[:]
        
        seed = sa_iter * 7919 + random.randint(0, 1000)
        new_edges, new_count = simulated_annealing(seed, base_edges, max_iters=200)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = new_edges
            current_best = new_edges[:]

    # Phase 3: Final polish - try to add any remaining edges
    if best_edges and time.time() < deadline - 5:
        adj = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        edge_set = set((min(u,v), max(u,v)) for u,v,_ in best_edges)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for u, v in all_edges:
                if (u,v) in edge_set:
                    continue
                impossible = False
                forbidden = set()
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
                                impossible = True
                                break
                            forbidden.add(ca)
                            forbidden.add(cb)
                            forbidden.add(cv)
                    if impossible:
                        break
                if not impossible:
                    valid_colors = [c for c in range(num_colors) if c not in forbidden]
                    if valid_colors:
                        c = valid_colors[0]
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edge_set.add((u,v))
                        best_edges.append([u, v, c])
                        best_count = len(best_edges)
                        improved = True

    return best_edges
