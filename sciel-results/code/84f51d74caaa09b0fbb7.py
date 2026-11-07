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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Domains as lists for easy manipulation
        domains = [list(l) for l in lists]
        
        # AC-3 propagation
        def ac3(domains):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    if v > u: # Add each edge once
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                if revise(domains, u, v):
                    if len(domains[u]) == 0:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        def revise(domains, i, j):
            revised = False
            for x in list(domains[i]):
                if not any(x != y for y in domains[j]):
                    domains[i].remove(x)
                    revised = True
            return revised

        # Backtracking search
        def backtrack(domains):
            # Find unassigned variable with smallest domain (MRV)
            unassigned = [i for i in range(n) if len(domains[i]) > 1]
            if not unassigned:
                # Check if all are assigned (size 1) and consistent
                # Actually, if we use domains of size 1 for assigned, 
                # we need to verify consistency one last time or handle it in propagation.
                # Standard approach: assign variables one by one.
                pass
            
            # If no unassigned, we are done? 
            # We need to distinguish between "unassigned" and "assigned".
            # Let's use a separate assignment array or treat domain size 1 as assigned.
            
            # Better: 
            # 1. Check for empty domain
            for i in range(n):
                if len(domains[i]) == 0:
                    return None
            
            # Find variable with smallest domain > 1
            unassigned = [i for i in range(n) if len(domains[i]) > 1]
            if not unassigned:
                # All domains have size 1. Verify consistency.
                for u in range(n):
                    cu = domains[u][0]
                    for v in adj[u]:
                        if domains[v][0] == cu:
                            return None
                return [d[0] for d in domains]
                
            v = min(unassigned, key=lambda i: len(domains[i]))
            
            # Try each value in domain
            for val in domains[v]:
                # Save state
                old_domains = [list(d) for d in domains]
                
                # Assign val to v
                domains[v] = [val]
                
                # Propagate
                if ac3(domains):
                    result = backtrack(domains)
                    if result is not None:
                        return result
                
                # Restore
                domains = old_domains
            
            return None

        if not ac3(domains):
            return None
        return backtrack(domains)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        if coloring is None:
            # Fallback: simple greedy (should not happen)
            coloring = [0] * n
            # Try to fix with a simple greedy pass
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for v in range(n):
                taken = {coloring[u] for u in adj[v]}
                for c in lists[v]:
                    if c not in taken:
                        coloring[v] = c
                        break
                else:
                    coloring[v] = lists[v][0]
        # Ensure types are Python int and list
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
