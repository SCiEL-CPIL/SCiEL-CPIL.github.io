"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by descending degree for better propagation
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        # color[v] = assigned color or -1 if unassigned
        # For each vertex, maintain the set of possible colors from its list
        # We use a queue for constraint propagation
        color = [-1] * n
        possible = [set(lists[v]) for v in range(n)]
        removed = [set() for _ in range(n)]  # colors removed from possible[v]
        
        queue = deque()
        
        # Initialize: if any vertex has only 1 possible color, queue it
        for v in range(n):
            if len(possible[v]) == 1:
                color[v] = next(iter(possible[v]))
                queue.append(v)
        
        # Constraint propagation
        while queue:
            v = queue.popleft()
            if color[v] == -1:
                continue
            cv = color[v]
            for u in adj[v]:
                if color[u] == -1 and cv in possible[u]:
                    # Remove cv from possible[u]
                    possible[u].discard(cv)
                    removed[u].add(cv)
                    if len(possible[u]) == 0:
                        # Dead end - need backtracking
                        # For now, return None to signal failure
                        return None
                    if len(possible[u]) == 1:
                        color[u] = next(iter(possible[u]))
                        queue.append(u)
                    elif len(possible[u]) == 2:
                        # Could queue for further propagation if needed, but keep it simple
                        pass
        
        # Check if all vertices are colored
        for v in range(n):
            if color[v] == -1:
                # Try to assign a color from remaining possibilities
                # Use a simple greedy with backtracking if needed
                # For robustness, let's try a more careful assignment
                return _backtrack_assign(n, adj, order, pos, color, possible, lists)
        
        # Verify the coloring
        for v in range(n):
            if color[v] not in lists[v]:
                return None
        for u, v in edges:
            if color[u] == color[v]:
                return None
        
        return color
    
    def _backtrack_assign(n, adj, order, pos, color, possible, lists):
        # Find uncolored vertices
        uncolored = [v for v in range(n) if color[v] == -1]
        if not uncolored:
            return color
        
        # Pick the vertex with the smallest domain
        v = min(uncolored, key=lambda x: len(possible[x]))
        
        # Try each possible color
        for c in list(possible[v]):
            # Check if c conflicts with any colored neighbor
            conflict = False
            for u in adj[v]:
                if color[u] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Assign c to v
            old_color = color[v]
            old_possible = set(possible[v])
            color[v] = c
            possible[v] = {c}
            
            # Propagate
            result = _propagate_and_solve(n, adj, order, pos, color, possible, lists, v)
            if result is not None:
                return result
            
            # Undo
            color[v] = old_color
            possible[v] = old_possible
        
        return None
    
    def _propagate_and_solve(n, adj, order, pos, color, possible, lists, start_v):
        # Simplified: just check if current state is valid and complete
        # This is a fallback; the main propagation should handle most cases
        from collections import deque
        
        queue = deque()
        for v in range(n):
            if color[v] != -1 and len(possible[v]) > 1:
                possible[v] = {color[v]}
            if color[v] == -1 and len(possible[v]) == 1:
                color[v] = next(iter(possible[v]))
                queue.append(v)
        
        while queue:
            v = queue.popleft()
            cv = color[v]
            for u in adj[v]:
                if color[u] == -1 and cv in possible[u]:
                    possible[u].discard(cv)
                    if len(possible[u]) == 0:
                        return None
                    if len(possible[u]) == 1:
                        color[u] = next(iter(possible[u]))
                        queue.append(u)
        
        # Check if all colored
        for v in range(n):
            if color[v] == -1:
                # Need more backtracking
                return None
        
        # Verify
        for v in range(n):
            if color[v] not in lists[v]:
                return None
        for u, v in edges_global:
            if color[u] == color[v]:
                return None
        
        return color

    # Global edges reference for the helper
    edges_global = []

    def solve_instance_wrapper(n, edges, lists):
        global edges_global
        edges_global = edges
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order by degree
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        color = [-1] * n
        possible = [set(lists[v]) for v in range(n)]
        
        # Initial propagation
        from collections import deque
        queue = deque()
        for v in range(n):
            if len(possible[v]) == 1:
                color[v] = next(iter(possible[v]))
                queue.append(v)
        
        while queue:
            v = queue.popleft()
            if color[v] == -1:
                continue
            cv = color[v]
            for u in adj[v]:
                if color[u] == -1 and cv in possible[u]:
                    possible[u].discard(cv)
                    if len(possible[u]) == 0:
                        return None
                    if len(possible[u]) == 1:
                        color[u] = next(iter(possible[u]))
                        queue.append(u)
        
        # If all colored, verify and return
        if all(color[v] != -1 for v in range(n)):
            for v in range(n):
                if color[v] not in lists[v]:
                    return None
            for u, v in edges:
                if color[u] == color[v]:
                    return None
            return color
        
        # Otherwise, use backtracking
        return _backtrack(n, adj, order, pos, color, possible, lists, edges)
    
    def _backtrack(n, adj, order, pos, color, possible, lists, edges):
        # Find uncolored vertex with smallest domain
        uncolored = [v for v in range(n) if color[v] == -1]
        if not uncolored:
            # All colored, verify
            for v in range(n):
                if color[v] not in lists[v]:
                    return None
            for u, v in edges:
                if color[u] == color[v]:
                    return None
            return color
        
        v = min(uncolored, key=lambda x: len(possible[x]))
        
        for c in sorted(possible[v]):
            # Check neighbor conflicts
            ok = True
            for u in adj[v]:
                if color[u] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            # Save state
            saved_color = color[v]
            saved_possible = [set(p) for p in possible]
            
            color[v] = c
            possible[v] = {c}
            
            # Propagate
            from collections import deque
            queue = deque([v])
            valid = True
            while queue and valid:
                u = queue.popleft()
                cu = color[u]
                if cu == -1:
                    continue
                for w in adj[u]:
                    if color[w] == -1 and cu in possible[w]:
                        possible[w].discard(cu)
                        if len(possible[w]) == 0:
                            valid = False
                            break
                        if len(possible[w]) == 1:
                            color[w] = next(iter(possible[w]))
                            queue.append(w)
            
            if valid:
                result = _backtrack(n, adj, order, pos, color, possible, lists, edges)
                if result is not None:
                    return result
            
            # Restore state
            color[v] = saved_color
            possible = saved_possible
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance_wrapper(n, edges, lists)
        if coloring is None:
            # Fallback to greedy
            coloring = _greedy_fallback(n, edges, lists)
        out.append(coloring)
    return out

def _greedy_fallback(n, edges, lists):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    colour = [None] * n
    for v in sorted(range(n), key=lambda u: -len(adj[u])):
        taken = {colour[u] for u in adj[v] if colour[u] is not None}
        free = [c for c in lists[v] if c not in taken]
        colour[v] = free[0] if free else lists[v][0]
    return colour

# EVOLVE-BLOCK-END
