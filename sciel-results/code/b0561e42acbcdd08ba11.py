def construct():
    import time
    import random
    from itertools import combinations

    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def count_paths_from_orientation(edges, orientation):
        """Count number of paths given an orientation.
        orientation: list of +1/-1 for each edge index
        +1 means edge[i]=(u,v) oriented u->v
        -1 means edge[i]=(u,v) oriented v->u
        """
        n = len(max(e for e in edges), default=0) + 1
        if not edges:
            return 0
        n = max(max(e) for e in edges) + 1
        
        in_deg = [0] * n
        out_deg = [0] * n
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                out_deg[u] += 1
                in_deg[v] += 1
            else:
                out_deg[v] += 1
                in_deg[u] += 1
        
        # Number of paths = sum over vertices of max(0, out_deg - in_deg)
        # which equals sum over vertices of max(0, in_deg - out_deg)
        # and equals half the sum of absolute imbalances
        total = 0
        for v in range(n):
            total += max(0, out_deg[v] - in_deg[v])
        return total

    def reconstruct_paths_from_orientation(edges, orientation, n):
        """Reconstruct actual paths from orientation."""
        out_adj = [[] for _ in range(n)]
        in_adj = [[] for _ in range(n)]
        
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                out_adj[u].append(v)
                in_adj[v].append(u)
            else:
                out_adj[v].append(u)
                in_adj[u].append(v)
        
        # Find starts: vertices with out_deg > in_deg
        starts = []
        for v in range(n):
            if len(out_adj[v]) > len(in_adj[v]):
                starts.append(v)
        
        paths = []
        visited = set()
        
        for start in starts:
            if not out_adj[start]:
                continue
            path = [start]
            current = start
            while out_adj[current]:
                # Find an unvisited outgoing edge
                next_v = None
                for v in out_adj[current]:
                    edge_key = (min(current, v), max(current, v))
                    if edge_key not in visited:
                        next_v = v
                        break
                if next_v is None:
                    break
                visited.add((min(current, next_v), max(current, next_v)))
                path.append(next_v)
                current = next_v
            if len(path) > 1:
                paths.append(path)
        
        # Handle remaining edges
        remaining = []
        for i, (u, v) in enumerate(edges):
            edge_key = (u, v)
            if edge_key not in visited:
                remaining.append((u, v))
        
        for u, v in remaining:
            paths.append([u, v])
        
        return paths

    def initial_orientation(edges, n, rng):
        """Create initial orientation by trying to make Eulerian-like structure."""
        m = len(edges)
        orientation = []
        
        # Greedy: try to orient to balance degrees
        # Start with all +1, then flip edges to reduce imbalance
        orientation = [1] * m
        
        # Simple local search from start
        for _ in range(200):
            changed = False
            for i in range(m):
                old_count = count_paths_from_orientation(edges, orientation)
                orientation[i] *= -1
                new_count = count_paths_from_orientation(edges, orientation)
                if new_count <= old_count:
                    changed = True
                    if new_count < old_count:
                        break
                else:
                    orientation[i] *= -1
            if not changed:
                break
        
        return orientation

    def local_search_orientation(edges, n, rng, max_iter=5000):
        """Local search to minimize path count via edge flips."""
        m = len(edges)
        orientation = [1] * m
        
        # First pass: greedy orient to minimize
        current_count = count_paths_from_orientation(edges, orientation)
        
        # Try all single flips
        for _ in range(max_iter):
            improved = False
            for i in range(m):
                orientation[i] *= -1
                new_count = count_paths_from_orientation(edges, orientation)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                    break
                else:
                    orientation[i] *= -1
            if not improved:
                break
        
        return orientation, current_count

    def two_opt_search(edges, n, rng, max_iter=2000):
        """2-opt style: try flipping pairs of edges."""
        m = len(edges)
        orientation = [1] * m
        
        # First do single-flip local search
        orientation, current_count = local_search_orientation(edges, n, rng, 5000)
        
        for _ in range(max_iter):
            improved = False
            # Try random pairs
            for attempt in range(10):
                i1 = rng.randint(0, m - 1)
                i2 = rng.randint(0, m - 1)
                if i1 == i2:
                    continue
                
                orientation[i1] *= -1
                orientation[i2] *= -1
                new_count = count_paths_from_orientation(edges, orientation)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                    break
                else:
                    orientation[i1] *= -1
                    orientation[i2] *= -1
            
            if not improved:
                # Try single flips again
                for i in range(m):
                    orientation[i] *= -1
                    new_count = count_paths_from_orientation(edges, orientation)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        break
                    else:
                        orientation[i] *= -1
                if not improved:
                    break
        
        return orientation, current_count

    def solve_instance(n, p, seed):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        
        m = len(edges)
        rng = random.Random(seed * 1000 + 42)
        
        best_orientation = None
        best_count = m  # worst case: each edge is its own path
        
        # Multiple restarts with different initial orientations
        for restart in range(5):
            # Different initial strategies
            if restart == 0:
                orientation = [1] * m
            elif restart == 1:
                orientation = [-1] * m
            elif restart == 2:
                orientation = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
            elif restart == 3:
                # Orient based on vertex ordering
                orientation = [1 if edges[i][0] < edges[i][1] else -1 for i in range(m)]
            else:
                orientation = [1 if rng.random() < 0.7 else -1 for _ in range(m)]
            
            # Local search
            orientation, count = two_opt_search(edges, n, rng, max_iter=1000)
            
            if count < best_count:
                best_count = count
                best_orientation = orientation[:]
            
            if time.time() > DEADLINE:
                break
        
        if best_orientation is None:
            best_orientation = [1] * m
        
        return reconstruct_paths_from_orientation(edges, best_orientation, n)

    # Solve all three instances
    results = []
    for n, p, seed in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > DEADLINE:
            # Emergency fallback: simple greedy
            edges = instance_graph(n, p, seed)
            if not edges:
                results.append([])
                continue
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            remaining = set(edges)
            paths = []
            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                start = max(range(n), key=lambda i: rem_deg[i])
                path = [start]
                used_v = {start}
                curr = start
                while True:
                    cands = [nb for nb in adj[curr] if (min(curr, nb), max(curr, nb)) in remaining and nb not in used_v]
                    if not cands:
                        break
                    cands.sort(key=lambda x: rem_deg[x], reverse=True)
                    next_v = cands[0]
                    path.append(next_v)
                    used_v.add(next_v)
                    remaining.remove((min(curr, next_v), max(curr, next_v)))
                    rem_deg[curr] -= 1
                    rem_deg[next_v] -= 1
                    curr = next_v
                paths.append(path)
            results.append(paths)
        else:
            results.append(solve_instance(n, p, seed))
    
    return results
