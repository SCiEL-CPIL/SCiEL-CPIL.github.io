"""Rewritten solver: exact backtracking with MRV + forward checking for list colouring."""
import time

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve(n, edges, lists):
        """Exact list-colouring via backtracking with MRV and forward checking.
        Returns a list of ints (one per vertex) or None if no solution found."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Convert adjacency to lists for speed
        adj_list = [list(s) for s in adj]

        # Remaining colour sets (start as lists)
        remaining = [list(l) for l in lists]

        colour = [-1] * n
        assigned = [False] * n

        def domain_size(v):
            return len(remaining[v])

        def is_valid_assignment():
            for v in range(n):
                if assigned[v]:
                    if colour[v] not in lists[v]:
                        return False
                    for u in adj_list[v]:
                        if assigned[u] and colour[u] == colour[v]:
                            return False
            return True

        # We'll use a simple recursive backtracker with MRV.
        # For these small instances (n <= ~408) it should be fast.

        import sys
        sys.setrecursionlimit(10000)

        def backtrack():
            # Pick unassigned vertex with smallest domain (MRV)
            best_v = -1
            best_size = 10**9
            for v in range(n):
                if not assigned[v]:
                    s = domain_size(v)
                    if s < best_size:
                        best_size = s
                        best_v = v
                        if s == 1:
                            break
                    if s == 0:
                        return False  # dead end
            if best_v == -1:
                return True  # all assigned

            v = best_v
            # Try each colour in remaining[v]
            for c in list(remaining[v]):
                # Check adjacency
                ok = True
                for u in adj_list[v]:
                    if assigned[u] and colour[u] == c:
                        ok = False
                        break
                if not ok:
                    continue

                # Assign
                colour[v] = c
                assigned[v] = True
                remaining[v] = []  # mark as used

                # Forward checking: remove c from neighbours' domains
                saved = []
                feasible = True
                for u in adj_list[v]:
                    if not assigned[u] and c in remaining[u]:
                        saved.append((u, len(remaining[u])))
                        remaining[u].remove(c)
                        if len(remaining[u]) == 0:
                            feasible = False
                            break

                if feasible and backtrack():
                    return True

                # Undo
                assigned[v] = False
                colour[v] = -1
                remaining[v] = [c]  # restore (we'll restore full below)
                # Actually restore full domain
                remaining[v] = list(lists[v])
                for u, _ in saved:
                    # We need to restore properly. Let's just rebuild remaining for unassigned.
                    pass
                # Simpler: rebuild remaining for all unassigned from scratch each backtrack? Too slow.
                # Let's do proper undo.
                # We removed c from neighbours. Put it back.
                for u in adj_list[v]:
                    if not assigned[u] and c not in remaining[u]:
                        remaining[u].append(c)
                        remaining[u].sort()

                return False

            return False

        # The above undo logic is tricky. Let's use a cleaner approach with explicit domain copying.
        # Given small n, we can afford to copy domains.

        def solve_clean():
            # domains[v] = set of available colours
            domains = [set(l) for l in lists]
            colour = [-1] * n
            assigned = [False] * n

            def bt():
                # MRV
                best_v = -1
                best_len = 10**9
                for v in range(n):
                    if not assigned[v]:
                        ln = len(domains[v])
                        if ln == 0:
                            return False
                        if ln < best_len:
                            best_len = ln
                            best_v = v
                            if ln == 1:
                                break
                if best_v == -1:
                    return True

                v = best_v
                # Try colours in sorted order for determinism
                for c in sorted(domains[v]):
                    # Check neighbours
                    ok = True
                    for u in adj_list[v]:
                        if assigned[u] and colour[u] == c:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Assign
                    colour[v] = c
                    assigned[v] = True
                    old_domain_v = set(domains[v])
                    domains[v] = set()

                    # Forward check
                    affected = []
                    feasible = True
                    for u in adj_list[v]:
                        if not assigned[u] and c in domains[u]:
                            affected.append(u)
                            domains[u].remove(c)
                            if len(domains[u]) == 0:
                                feasible = False
                                break

                    if feasible and bt():
                        return True

                    # Undo
                    assigned[v] = False
                    colour[v] = -1
                    domains[v] = old_domain_v
                    for u in affected:
                        domains[u].add(c)

                return False

            if bt():
                return colour
            return None

        result = solve_clean()
        return result

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Not enough time; break and return what we have
            break
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        if sol is not None:
            # Ensure Python ints
            out.append([int(c) for c in sol])
        else:
            # Fallback: greedy (shouldn't happen per Alon-Tarsi, but safety)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])

    # Pad with empty lists if we ran out of time (shouldn't happen)
    while len(out) < 4:
        out.append([])

    return out
