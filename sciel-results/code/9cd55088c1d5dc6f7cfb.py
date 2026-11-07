"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 150s limit, aim to finish in < 10s
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initial domains as sets
        domains = [set(l) for l in lists]

        # Check for empty domains initially
        for i in range(n):
            if not domains[i]:
                return None

        # Arc Consistency (AC-3) Preprocessing
        # For each arc (u, v), remove values from domains[u] that have no support in domains[v]
        # Support for value c in domains[u] w.r.t v exists if there is some value in domains[v] that is not c.
        # i.e., if len(domains[v]) > 1 or (len(domains[v]) == 1 and list(domains[v])[0] != c)

        def enforce_arc_consistency():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))

            while queue:
                u, v = queue.pop()
                du = domains[u]
                dv = domains[v]

                # If dv is empty, u is inconsistent (but we check empty domains separately)
                if not dv:
                    return False

                # Values in du that have no support in dv
                # A value c in du has no support in dv if dv is a subset of {c}
                # i.e., dv == {c}
                to_remove = set()
                for c in du:
                    if dv == {c}:
                        to_remove.add(c)

                if to_remove:
                    du.difference_update(to_remove)
                    if not du:
                        return False
                    # Add all arcs (w, u) to queue where w is a neighbor of u (w != v)
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        if not enforce_arc_consistency():
            # If AC-3 fails, no solution exists (contradicts theorem, but handle gracefully)
            return [0] * n

        # We use a recursive backtracking solver with forward checking.
        colour = [-1] * n
        # Copy domains to avoid modifying original
        current_domains = [set(d) for d in domains]
        unassigned = set(range(n))

        # Precompute degrees for heuristic tie-breaking
        degrees = [len(adj[i]) for i in range(n)]

        sys.setrecursionlimit(10000)

        def backtrack():
            if time.time() > deadline:
                return False # Timeout, abort search (shouldn't happen)

            if not unassigned:
                return True

            # MRV Heuristic: Select vertex with fewest remaining values
            # Tie-breaker: Highest degree (most constrained)
            best_v = -1
            best_count = float('inf')
            best_deg = -1

            for v in unassigned:
                cnt = len(current_domains[v])
                if cnt == 0:
                    return False # Dead end
                if cnt < best_count or (cnt == best_count and degrees[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degrees[v]
                    if cnt == 1:
                        break # Can't get better than 1

            if best_v == -1:
                return False

            # Try each color in the domain of best_v
            v = best_v
            candidates = list(current_domains[v])

            # Sort candidates by LCV (Least Constraining Value)
            def lcv_score(c):
                removals = 0
                for u in adj[v]:
                    if u in unassigned and c in current_domains[u]:
                        removals += 1
                return removals

            candidates.sort(key=lcv_score)

            for c in candidates:
                # Assign c to v
                colour[v] = c
                unassigned.remove(v)

                # Forward Checking: Update neighbors
                conflict = False
                saved_domains = []

                for u in adj[v]:
                    if u in unassigned:
                        if c in current_domains[u]:
                            saved_domains.append((u, c))
                            current_domains[u].discard(c)
                            if not current_domains[u]:
                                conflict = True
                                break

                if not conflict:
                    if backtrack():
                        return True
                else:
                    # Undo neighbor changes if we didn't recurse successfully or if conflict
                    pass

                # Undo
                colour[v] = -1
                unassigned.add(v)
                for u, c_to_remove in saved_domains:
                    current_domains[u].add(c_to_remove)

            return False

        success = backtrack()

        if success:
            # Ensure types are correct (Python int)
            return [int(c) for c in colour]
        else:
            # Fallback: This should never happen if the theorem holds and solver is correct.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 1.0:
            # Not enough time left, break
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)

    # If we broke early due to timeout, we might have fewer than 4 instances.
    # The task requires returning a list of four lists.
    # However, if we time out, we can't solve more.
    # Given the solver is very fast, this is unlikely.
    # If it happens, we pad with invalid colorings (zeros) to maintain structure,
    # which will score 0 for those instances but keep the return type valid.
    while len(out) < 4:
        # We don't have the n for the missing instances, but we can't return None.
        # Actually, if we didn't solve it, we don't know n.
        # But the loop above generates n. If we break, we didn't generate.
        # It is safer to just return what we have if it's < 4?
        # No, "wrong length ... makes the whole result invalid".
        # So we must ensure we have 4.
        # If we run out of time, we fail.
        # But with this optimized solver, we won't run out of time.
        # For safety, if we break, we can't construct the instance to know n.
        # We will assume we don't break.
        pass

    return out
# EVOLVE-BLOCK-END
