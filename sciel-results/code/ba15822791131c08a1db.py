"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domains: list of lists (tuples for immutability in snapshots, but sets are faster for ops)
        # We will use sets for domains and snapshot them for backtracking.
        domains = [set(lists[v]) for v in range(n)]
        
        # Check for initial empty domains
        for v in range(n):
            if not domains[v]:
                return None

        # AC-3 Implementation that modifies domains in place.
        # Returns False if a domain becomes empty.
        def ac3(domains, adj, n, queue=None):
            if queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                # Constraint: u != v
                # Remove c from domains[u] if domains[v] is a subset of {c}
                # i.e., if domains[v] == {c}, then c cannot be in domains[u].
                # More generally, if domains[v] is empty, we already failed.
                # If domains[v] has 1 element, say x, then x cannot be in domains[u].
                # If domains[v] has >1 elements, no single color in domains[u] is blocked by v alone.
                
                if len(domains[v]) == 1:
                    only_color = next(iter(domains[v]))
                    if only_color in domains[u]:
                        domains[u].discard(only_color)
                        if not domains[u]:
                            return False
                        # Add neighbors of u (except v) to queue
                        for z in adj[u]:
                            if z != v:
                                queue.append((z, u))
            return True

        # Initial AC-3 pass
        if not ac3(domains, adj, n):
            return None

        assigned = [None] * n
        # Stack of snapshots for backtracking
        # Each snapshot is a list of sets (copies)
        
        def get_mrv_var():
            min_size = float('inf')
            min_degree = float('inf')
            var = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    # Count unassigned neighbors for degree heuristic
                    degree = sum(1 for u in adj[v] if assigned[u] is None)
                    if size < min_size or (size == min_size and degree < min_degree):
                        min_size = size
                        min_degree = degree
                        var = v
                        if size <= 1:
                            break
            return var

        def backtrack():
            if time.monotonic() > deadline:
                return False
            
            var = get_mrv_var()
            if var == -1:
                return True # All assigned
            
            # LVC Heuristic: Order values by how few constraints they impose on neighbors
            values = list(domains[var])
            def value_score(c):
                count = 0
                for u in adj[var]:
                    if assigned[u] is None and c in domains[u]:
                        count += 1
                return count
            
            values.sort(key=value_score)
            
            for c in values:
                if c not in domains[var]:
                    continue
                
                # Take snapshot of current domains
                # Copying sets of small ints is fast
                snapshot = [set(d) for d in domains]
                
                # Assign
                assigned[var] = c
                domains[var].clear()
                
                # Forward Checking + AC-3
                # Remove c from neighbors' domains
                ok = True
                queue = deque()
                for u in adj[var]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            domains[u].discard(c)
                            if not domains[u]:
                                ok = False
                                break
                            # Add neighbors of u to queue for AC-3
                            for z in adj[u]:
                                if z != var and assigned[z] is None:
                                    queue.append((z, u))
                
                if ok:
                    # Run AC-3 on the affected parts
                    if not ac3(domains, adj, n, queue):
                        ok = False
                
                if ok:
                    if backtrack():
                        return True
                
                # Undo: Restore snapshot
                for i in range(n):
                    domains[i] = snapshot[i]
                assigned[var] = None
                
                if time.monotonic() > deadline:
                    return False
            
            return False

        if backtrack():
            return [int(assigned[v]) for v in range(n)]
        else:
            # Fallback: This should not happen for colorable instances.
            # Return a random valid assignment from lists (may violate edges)
            return [int(lists[v][0]) for v in range(n)]

    start = time.monotonic()
    deadline = start + 115.0  # Safety margin before 120s limit

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.monotonic() > deadline - 5:
            n, edges, lists = instance(splits, seed)
            out.append([int(lists[v][0]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        
        # Validate result strictly before adding
        if len(result) != n:
            result = [int(lists[v][0]) for v in range(n)]
        else:
            valid = True
            for i, c in enumerate(result):
                if not isinstance(c, int) or isinstance(c, bool):
                    valid = False
                    break
                if c not in lists[i]:
                    valid = False
                    break
            
            if valid:
                for u, v in edges:
                    if result[u] == result[v]:
                        valid = False
                        break
            
            if not valid:
                result = [int(lists[v][0]) for v in range(n)]
        
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
