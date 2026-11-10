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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bipartition: BFS from vertex 0
        color_side = [-1] * n
        color_side[0] = 0
        queue = [0]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if color_side[v] == -1:
                    color_side[v] = 1 - color_side[u]
                    queue.append(v)

        for i in range(n):
            if color_side[i] == -1:
                color_side[i] = 0

        # Copy lists to domains
        domains = [list(l) for l in lists]
        assignment = [None] * n

        # AC-3 implementation
        def revise(x, y):
            revised = False
            dx = domains[x]
            dy = domains[y]
            # Remove values from dx that conflict with dy
            # Since dy is a set of allowed colors, we remove any c in dx if c is in dy
            # Wait, constraint is c_x != c_y.
            # So if c is in dx and c is in dy, it is NOT necessarily removed.
            # It is removed only if we KNOW c is the *only* value in dy? No.
            # Arc consistency: c in dx is consistent if there exists d in dy such that c != d.
            # If dy == {c}, then c in dx is inconsistent.
            # If len(dy) == 1 and dy[0] == c, remove c from dx.
            if len(dy) == 1:
                val = dy[0]
                if val in dx:
                    dx.remove(val)
                    revised = True
            return revised

        def ac3(domains):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            while queue:
                u, v = queue.popleft()
                if revise(u, v):
                    if not domains[u]:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        # Initial AC-3
        if not ac3(domains):
            return None # Should not happen

        # Backtracking with MRV and Forward Checking
        def backtrack(assignment, domains):
            # Find unassigned vertex with smallest domain (MRV)
            unassigned = [i for i in range(n) if assignment[i] is None]
            if not unassigned:
                return True
            
            # MRV: pick vertex with smallest domain size
            min_size = float('inf')
            var = -1
            for v in unassigned:
                if len(domains[v]) < min_size:
                    min_size = len(domains[v])
                    var = v
            
            if min_size == 0:
                return False

            # LCV: order values by least constraining value
            # For each color c in domains[var], count how many neighbors lose how many options
            # Simplified LCV: try colors that are less frequent in neighbors' domains?
            # Or just try in order. For small domains, order doesn't matter much.
            values = domains[var][:]
            
            for c in values:
                # Check if c is still in domain (it should be)
                if c not in domains[var]:
                    continue
                
                # Assign c to var
                assignment[var] = c
                old_domains = [list(d) for d in domains]
                
                # Forward Checking: remove c from neighbors' domains
                valid = True
                for u in adj[var]:
                    if assignment[u] is None:
                        if c in domains[u]:
                            domains[u].remove(c)
                            if not domains[u]:
                                valid = False
                                break
                
                if valid:
                    # Optional: Run AC-3 on affected neighbors? 
                    # For speed, just FC is often enough for these sizes.
                    if backtrack(assignment, domains):
                        return True
                
                # Undo
                assignment[var] = None
                domains = old_domains

            return False

        if backtrack(assignment, domains):
            return [int(c) for c in assignment]
        else:
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        if coloring is None:
             # Fallback to naive if AC fails (shouldn't happen)
             coloring = [lists[v][0] for v in range(n)]
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
