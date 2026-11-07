"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import time as _time

    deadline = _time.monotonic() + 130.0

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

    def solve_csp(n, edges, color_lists):
        """Optimized Backtracking CSP solver with forward checking and MRV."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees for greedy fallback if needed
        degrees = [len(a) for a in adj]
        
        domains = [set(color_lists[v]) for v in range(n)]
        assignment = [None] * n
        node_count = 0
        max_nodes = 20000  # Safety limit to prevent pathological slowness

        def forward_check(v, c):
            changes = []
            for u in adj[v]:
                if assignment[u] is None:
                    dom_u = domains[u]
                    if c in dom_u:
                        new_dom = dom_u - {c}
                        if not new_dom:
                            for (uu, old_dom) in changes:
                                domains[uu] = old_dom
                            return None
                        changes.append((u, dom_u))
                        domains[u] = new_dom
            return changes

        def undo_forward_check(changes):
            for (u, old_dom) in changes:
                domains[u] = old_dom

        def get_mrv_vertex():
            min_size = 4 # Max domain size is 3
            best = -1
            for v in range(n):
                if assignment[v] is None:
                    sz = len(domains[v])
                    if sz == 0:
                        return v
                    if sz < min_size:
                        min_size = sz
                        best = v
                        if sz == 1:
                            break
            return best

        def backtrack(depth):
            nonlocal node_count
            if _time.monotonic() > deadline or node_count > max_nodes:
                return None
            
            node_count += 1
            v = get_mrv_vertex()
            if v == -1:
                return assignment[:]

            d = domains[v]
            # Heuristic: Least Constraining Value (LCV)
            # Sort by number of eliminations in neighbors
            def lcv_score(c):
                count = 0
                for u in adj[v]:
                    if assignment[u] is None and c in domains[u]:
                        count += 1
                return count
            
            sorted_colors = sorted(d, key=lcv_score)

            for c in sorted_colors:
                assignment[v] = c
                changes = forward_check(v, c)
                if changes is not None:
                    result = backtrack(depth + 1)
                    if result is not None:
                        return result
                undo_forward_check(changes if changes is not None else [])
                assignment[v] = None

            return None

        result = backtrack(0)
        if result is not None:
            return result

        # Fallback: Greedy with degree ordering
        colour = [None] * n
        order = sorted(range(n), key=lambda u: -degrees[u])
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in color_lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                # Should not happen if graph is colorable, but pick first to avoid None
                colour[v] = color_lists[v][0]
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if _time.monotonic() > deadline:
            # Not enough time; return greedy fallback for remaining
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
            continue
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
