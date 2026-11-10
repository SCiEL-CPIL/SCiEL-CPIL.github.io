"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Copy lists to mutable domains
        domains = [list(l) for l in lists]
        
        # AC-3 Arc Consistency
        # Queue of arcs (u, v)
        queue = []
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v):
            # Remove values from domains[u] that have no support in domains[v]
            revised = False
            to_remove = []
            for c in domains[u]:
                # Check if there is any color in domains[v] different from c
                # Since colors are small (0..4), we can just check
                supported = False
                for c2 in domains[v]:
                    if c != c2:
                        supported = True
                        break
                if not supported:
                    to_remove.append(c)
            
            if to_remove:
                revised = True
                for c in to_remove:
                    domains[u].remove(c)
                # If domain becomes empty, failure
                if not domains[u]:
                    return False
                # Add arcs (w, u) for all w in adj[u] except v
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
            return True

        while queue:
            u, v = queue.pop(0)
            if not revise(u, v):
                return None # Failure
        
        # If we reach here, domains are arc-consistent
        # Use Backtracking with Forward Checking
        assignment = [None] * n
        
        # Heuristic: Order vertices by smallest remaining domain (MRV)
        # Since domains are static after AC-3 for the initial state, 
        # we can use static order based on initial domain size or degree.
        # For dynamic MRV, we'd need to re-order at each step.
        # Given small n, static order by degree is fine, or we can implement dynamic.
        
        # Let's use dynamic MRV for better performance
        def backtrack(unassigned):
            if not unassigned:
                return True
            
            # Choose variable with Minimum Remaining Values
            # Tie-break by degree
            v = min(unassigned, key=lambda x: (len(domains[x]), -len(adj[x])))
            
            # Try each color in the domain
            # Order colors by Least Constraining Value (LCV) - optional, skip for simplicity
            for c in domains[v]:
                # Check consistency with already assigned neighbors
                valid = True
                for u in adj[v]:
                    if assignment[u] is not None and assignment[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign v = c
                assignment[v] = c
                
                # Forward Check: Prune domains of unassigned neighbors
                # We need to save the state of domains to revert
                # Since n is small, we can copy domains
                # Or we can implement a trail.
                # Given the small size (n~400), copying is O(n*3) which is fine.
                old_domains = [list(d) for d in domains]
                
                # Prune neighbors
                fail = False
                for u in adj[v]:
                    if assignment[u] is None:
                        if c in domains[u]:
                            domains[u].remove(c)
                            if not domains[u]:
                                fail = True
                                break
                
                if not fail:
                    new_unassigned = [u for u in unassigned if u != v]
                    if backtrack(new_unassigned):
                        return True
                
                # Revert domains
                domains[:] = old_domains
                assignment[v] = None
            
            return False

        # Initial unassigned set
        unassigned = list(range(n))
        
        if backtrack(unassigned):
            return [int(c) for c in assignment]
        else:
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        # Ensure all values are Python ints
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
