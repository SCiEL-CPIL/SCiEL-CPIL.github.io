"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    DEADLINE = 145.0  # safety margin before 150s kill
    start_time = time.time()

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def solve_csp(n, adj, lists):
        """
        Systematic backtracking with MRV + forward checking.
        Returns a valid colouring list or None.
        """
        # Domain for each vertex: list of allowed colours
        domains = [set(c) for c in lists]

        # Order vertices by degree (descending) for initial ordering, but MRV will dominate
        # We'll use recursive backtracking with MRV heuristic

        # Precompute degree for tie-breaking
        degree = [len(adj[v]) for v in range(n)]

        def get_mrv_vertex(assigned):
            """Find unassigned vertex with minimum remaining values (MRV)."""
            best_v = -1
            best_count = float('inf')
            for v in range(n):
                if v not in assigned:
                    cnt = len(domains[v])
                    if cnt == 0:
                        return v  # fail
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
            return best_v

        def forward_check(v, c, assigned):
            """
            Assign colour c to vertex v, then reduce domains of unassigned neighbors.
            Returns (success, undo_info) where undo_info is list of (neighbor, removed_colors)
            """
            assigned[v] = c
            undo_info = []
            for u in adj[v]:
                if u not in assigned:
                    if c in domains[u]:
                        domains[u].discard(c)
                        undo_info.append((u, c))
                        if len(domains[u]) == 0:
                            # Backtrack immediately
                            for uu, cc in undo_info:
                                domains[uu].add(cc)
                            del assigned[v]
                            return False, None
            return True, undo_info

        def backtrack(assigned):
            if len(assigned) == n:
                return True

            v = get_mrv_vertex(assigned)
            if v == -1:
                return True

            if len(domains[v]) == 0:
                return False

            # Try colours in the domain
            for c in sorted(domains[v]):
                ok, undo_info = forward_check(v, c, assigned)
                if ok:
                    if backtrack(assigned):
                        return True
                    # Undo
                    for u, cc in undo_info:
                        domains[u].add(cc)
                    del assigned[v]

            return False

        assigned = {}
        if backtrack(assigned):
            return [assigned[v] for v in range(n)]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline
        if time.time() > DEADLINE - 5:
            # Not enough time, return what we have (shouldn't happen for 4 instances)
            break

        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        # Try CSP solver
        colouring = solve_csp(n, adj, lists)

        if colouring is None:
            # Fallback: greedy (shouldn't happen per Alon-Tarsi, but just in case)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            colouring = colour

        # Ensure all values are Python int
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    # Ensure we have exactly 4 results
    while len(out) < 4:
        out.append([])

    return out
# EVOLVE-BLOCK-END
