"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # MRV + Forward Checking backtracking
        # We'll use a recursive approach with domains
        domains = [list(l) for l in lists]
        
        # Pre-compute a good ordering: use dynamic MRV
        # But for efficiency, we can also try a static ordering first based on degree
        # Let's use a recursive solver that picks the vertex with the fewest remaining values
        
        # To avoid deep recursion issues, we use an iterative or careful recursive approach
        # Given n is at most 8 + 4*100 = 408, recursion depth could be an issue if we go deep,
        # but Python's default limit is 1000, so we should be fine.
        
        # We'll implement a standard backtracking with MRV and forward checking.
        
        # For speed, we can also try a greedy pass first, and only backtrack if it fails.
        # But the knowledge says backtracking guarantees success. Let's just do backtracking.
        
        # Optimization: Use a simple heuristic to order vertices initially?
        # Actually, dynamic MRV is usually best.
        
        # Let's implement the solver.
        
        # To handle potential performance issues, we can add a timeout or limit, 
        # but the task says 120s for 4 instances, so ~30s each. 
        # These graphs are sparse and bipartite, so backtracking should be very fast.
        
        # We'll use a stack-based or recursive approach. Recursive is cleaner.
        
        # We need to be careful with the domain updates (forward checking).
        
        def backtrack(idx):
            # Find the unassigned vertex with the fewest available values (MRV)
            # We need to track which vertices are assigned.
            # For efficiency, we can maintain a list of unassigned vertices and their current domain sizes.
            # But a simple scan is O(n) per step, which is fine for n~400.
            
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if domains[v] is not None:  # Unassigned
                    if len(domains[v]) < min_size:
                        min_size = len(domains[v])
                        min_vertex = v
                        if min_size == 0:
                            return False  # Dead end
                        if min_size == 1:
                            break  # Can't get better than 1
            
            if min_vertex == -1:
                return True  # All assigned
            
            v = min_vertex
            saved_domain = domains[v][:]  # Save for backtracking
            
            for color in domains[v]:
                # Try assigning color to v
                domains[v] = None  # Mark as assigned
                # Actually, we need to store the assigned color somewhere.
                # Let's use a separate array for assigned colors.
                
                # Wait, the current design mixes domains and assignment.
                # Let's refactor:
                # domains[v] is a list of available colors if unassigned, None if assigned.
                # assigned[v] is the color if assigned, -1 if not.
                
                # Let's restart the design of the solver state.
                pass

        # Let's write a cleaner version.
        
        assigned = [-1] * n
        domains = [list(l) for l in lists]
        
        def is_valid(v, color):
            for u in adj[v]:
                if assigned[u] == color:
                    return False
            return True

        def backtrack():
            # Find MRV vertex
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if assigned[v] == -1:
                    # Count valid colors
                    valid_colors = [c for c in domains[v] if is_valid(v, c)]
                    # We need to update domains for forward checking or just check on the fly?
                    # Checking on the fly is simpler but slower.
                    # Forward checking is better.
                    
                    # Let's stick to a simpler backtracking first:
                    # Just pick the first unassigned vertex in a static order (e.g., degree descending)
                    # and try colors.
                    # This might be slow if the order is bad.
                    
                    # Let's use static order: degree descending.
                    # And simple backtracking.
                    pass
            
            # Actually, let's use the static order approach first, as it's simpler and likely fast enough.
            # Order vertices by degree descending.
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            
            def backtrack_static(idx):
                if idx == n:
                    return True
                v = order[idx]
                for color in domains[v]:
                    if is_valid(v, color):
                        assigned[v] = color
                        if backtrack_static(idx + 1):
                            return True
                        assigned[v] = -1
                return False

            return backtrack_static(0)

        if backtrack():
            return assigned
        else:
            # Should not happen for these instances, but return something
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)
    return out
# EVOLVE-BLOCK-END
