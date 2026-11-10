"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import deque

    _deadline = time.time() + 140  # safety margin before 150s kill

    def instance(splits, seed, palette=5):
        rng = random.Random(seed)
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
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Domains as lists (initially the given lists)
        domains = [list(lists[i]) for i in range(n)]

        # AC-3: arc consistency preprocessing
        def ac3(domains, adj):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    if v > u:  # avoid duplicates
                        queue.append((u, v))
                        queue.append((v, u))
            while queue:
                u, v = queue.popleft()
                # Check if any value in domain[u] has no support in domain[v]
                removed = False
                new_dom_u = []
                for c in domains[u]:
                    if any(c != cv for cv in domains[v]):
                        new_dom_u.append(c)
                if len(new_dom_u) < len(domains[u]):
                    removed = True
                    domains[u] = new_dom_u
                    if not new_dom_u:
                        return False
                    # Add all neighbors of u (except v) to queue
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        if not ac3(domains, adj):
            return None

        # Backtracking with MRV and forward checking
        # Use iterative deepening or just straightforward backtracking with good heuristics
        assigned = [None] * n
        domains_copy = [list(d) for d in domains]

        def backtrack(assigned, domains):
            # Find unassigned vertex with MRV
            min_count = float('inf')
            min_vertex = -1
            for i in range(n):
                if assigned[i] is None:
                    if len(domains[i]) < min_count:
                        min_count = len(domains[i])
                        min_vertex = i
                        if min_count == 1:
                            break
            if min_vertex == -1:
                return True  # All assigned

            # Try each color in the domain
            for c in domains[min_vertex]:
                # Check if c conflicts with assigned neighbors
                conflict = False
                for nb in adj[min_vertex]:
                    if assigned[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                # Forward checking: see if any unassigned neighbor loses all options
                assigned[min_vertex] = c
                backup = []
                ok = True
                for nb in adj[min_vertex]:
                    if assigned[nb] is None and c in domains[nb]:
                        backup.append((nb, len(domains[nb])))
                        domains[nb] = [x for x in domains[nb] if x != c]
                        if not domains[nb]:
                            ok = False
                            break

                if ok:
                    if backtrack(assigned, domains):
                        return True

                # Undo
                assigned[min_vertex] = None
                for nb, old_len in backup:
                    # Restore domains[nb] - we need to restore properly
                    # This is tricky; let's use a different approach
                    pass

            return False

        # The above forward checking undo is complex. Let's use a cleaner approach:
        # Store full state or use a simpler backtracking with just MRV and no forward checking undo complexity.

        # Better approach: use a proper backtracking with domain copying for small problems
        # Since n is at most 8 + 4*100 = 408, and lists are size 3, this should be fine with good heuristics.

        # Let's use a cleaner recursive backtracker
        def solve_bt(domains, assigned):
            # Find MRV unassigned
            min_count = float('inf')
            min_vertex = -1
            for i in range(n):
                if assigned[i] is None:
                    cnt = len(domains[i])
                    if cnt == 0:
                        return False
                    if cnt < min_count:
                        min_count = cnt
                        min_vertex = i
            if min_vertex == -1:
                return True

            # Degree ordering as tiebreaker: prefer higher degree
            best_candidates = []
            for i in range(n):
                if assigned[i] is None and len(domains[i]) == min_count:
                    best_candidates.append(i)
            # Sort by degree descending
            best_candidates.sort(key=lambda x: -len(adj[x]))

            for v in best_candidates:
                for c in domains[v]:
                    # Check conflict with assigned neighbors
                    if any(assigned[nb] == c for nb in adj[v]):
                        continue

                    assigned[v] = c
                    # Try to propagate: remove c from unassigned neighbors' domains
                    affected = []
                    ok = True
                    for nb in adj[v]:
                        if assigned[nb] is None and c in domains[nb]:
                            affected.append((nb, c, len(domains[nb])))
                            domains[nb] = [x for x in domains[nb] if x != c]
                            if not domains[nb]:
                                ok = False
                                break

                    if ok and solve_bt(domains, assigned):
                        return True

                    # Undo
                    assigned[v] = None
                    for nb, c_removed, old_len in affected:
                        # Restore c to domains[nb] if it was removed
                        if c_removed not in domains[nb]:
                            domains[nb].append(c_removed)
                            domains[nb].sort()

            return False

        assigned = [None] * n
        if solve_bt(domains, assigned):
            return [int(x) for x in assigned]
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline:
            # Fill remaining with dummy (will score 0 but keep valid structure)
            # Actually we need to return something valid in structure
            # Let's just return empty lists or try a quick greedy
            n, edges, lists = instance(splits, seed)
            # Quick greedy as fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue

        n, edges, lists = instance(splits, seed)
        result = solve_list_coloring(n, edges, lists)
        if result is None:
            # Fallback to greedy
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            result = [int(c) for c in colour]
        out.append([int(c) for c in result])

    return out
# EVOLVE-BLOCK-END
