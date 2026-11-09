"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys

    # Increase recursion limit for backtracking
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)

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
        """Solve list colouring via AC-3 preprocessing + backtracking with MRV + forward checking."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Convert lists to bitmask (5 colours, bits 0-4)
        domain = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domain[v] = mask

        # AC-3 arc consistency preprocessing
        from collections import deque
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            u, v = queue.popleft()
            # Check if domain[u] has support in domain[v]
            # For each colour in domain[u], there must be a different colour in domain[v]
            # Since colours are just "different", support exists iff domain[v] has at least one bit
            # that is not the same as the chosen colour in domain[u]
            # More precisely: for each colour c in domain[u], we need some c' in domain[v] with c' != c
            # This fails only if domain[v] is a subset of {c} for some c in domain[u]
            # i.e., domain[v] has exactly one bit and that bit is in domain[u]
            if domain[v] == 0:
                return None  # already failed
            # domain[u] loses colour c if domain[v] has no support for c
            # c has no support in v iff domain[v] is a subset of {c}
            # i.e., domain[v] is a power of 2 and that power of 2 is in domain[u]
            if domain[v] & (domain[v] - 1) == 0:  # domain[v] has exactly one colour
                single_bit = domain[v]
                if domain[u] & single_bit:
                    # colour single_bit in domain[u] has no support in v
                    domain[u] &= ~single_bit
                    if domain[u] == 0:
                        return None
                    # Re-add neighbors of u (except v) to queue
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))

        # Colour assignment: -1 means unassigned
        colour = [-1] * n
        assigned = [False] * n

        # Precompute degree for tie-breaking
        degree = [len(adj[v]) for v in range(n)]

        def pick_next():
            """MRV: pick unassigned vertex with fewest available colours; tie-break by degree."""
            best_v = -1
            best_count = 999
            best_deg = -1
            for v in range(n):
                if not assigned[v]:
                    cnt = bin(domain[v]).count('1')
                    if cnt == 0:
                        return -1  # dead end
                    if cnt < best_count or (cnt == best_count and degree[v] > best_deg):
                        best_count = cnt
                        best_deg = degree[v]
                        best_v = v
            return best_v

        def backtrack():
            if time.time() > deadline:
                return False
            v = pick_next()
            if v == -1:
                # Check if all assigned
                for i in range(n):
                    if not assigned[i]:
                        return False
                return True

            # Try each colour in domain[v]
            mask = domain[v]
            while mask:
                bit = mask & (-mask)
                c = bit.bit_length() - 1
                mask -= bit

                # Assign colour c to v
                colour[v] = c
                assigned[v] = True

                # Forward checking: update domains of neighbours
                saved_domains = []
                ok = True
                for nb in adj[v]:
                    if not assigned[nb]:
                        saved_domains.append((nb, domain[nb]))
                        if domain[nb] & bit:
                            domain[nb] &= ~bit
                            if domain[nb] == 0:
                                ok = False
                                break

                if ok and backtrack():
                    return True

                # Undo
                colour[v] = -1
                assigned[v] = False
                for nb, old_dom in saved_domains:
                    domain[nb] = old_dom

            return False

        result = backtrack()
        if result:
            return [int(c) for c in colour]
        return None

    out = []
    deadline = time.time() + 140.0  # safety margin before 150s kill

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            break
        n, edges, lists = instance(splits, seed)
        sol = solve_csp(n, edges, lists)
        if sol is None:
            # Fallback: should not happen per Alon-Tarsi, but guard
            sol = [0] * n
        out.append(sol)

    sys.setrecursionlimit(old_limit)

    # Ensure types are correct: list of lists of Python ints
    final_out = []
    for coloring in out:
        final_out.append([int(c) for c in coloring])
    return final_out
# EVOLVE-BLOCK-END
