"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def is_full(colour, n, edges, lists):
        """Check if colouring is valid: correct length, types, list constraints, edge constraints."""
        if not isinstance(colour, (list, tuple)):
            return False
        if len(colour) != n:
            return False
        for i, c in enumerate(colour):
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    def valid_count(colour, n, edges, lists):
        """Count how many constraints are satisfied (for local search)."""
        # Check list membership
        for i, c in enumerate(colour):
            if not isinstance(c, int) or isinstance(c, bool) or c not in lists[i]:
                return 0
        # Count violated edges
        violated = 0
        for u, v in edges:
            if colour[u] == colour[v]:
                violated += 1
        return len(edges) - violated

    def backtrack_color(n, edges, lists, deadline):
        """Backtracking with MRV and forward checking."""
        adj = build_adj(n, edges)
        # Degree ordering for tie-breaking
        degree = [len(adj[i]) for i in range(n)]

        # Order vertices by degree descending for initial ordering
        order = sorted(range(n), key=lambda v: -degree[v])

        colour = [-1] * n
        # domain[v] = set of available colours for v
        domain = [set(lists[v]) for v in range(n)]

        # We'll use a stack-based approach with MRV
        # For efficiency, we process in a smart order

        # Use recursive backtracking with MRV
        sys.setrecursionlimit(100000)

        def solve():
            # Find unassigned vertex with minimum domain (MRV)
            best_v = -1
            best_size = float('inf')
            for v in range(n):
                if colour[v] == -1:
                    if len(domain[v]) < best_size:
                        best_size = len(domain[v])
                        best_v = v
                        if best_size == 0:
                            return False
            if best_v == -1:
                return True  # All assigned

            v = best_v
            # Try colours in order of least constraining value
            for c in sorted(domain[v], key=lambda c: -sum(1 for u in adj[v] if colour[u] == -1 and c in domain[u])):
                colour[v] = c
                # Forward check: remove c from neighbours' domains
                saved = []
                ok = True
                for u in adj[v]:
                    if colour[u] == -1 and c in domain[u]:
                        domain[u].discard(c)
                        saved.append((u, c))
                        if len(domain[u]) == 0:
                            ok = False
                            break
                if ok:
                    if solve():
                        return True
                # Undo
                colour[v] = -1
                for u, c2 in saved:
                    domain[u].add(c2)
            return False

        if solve():
            return [int(c) for c in colour]
        return None

    def local_search(n, edges, lists, deadline):
        """Local search / repair approach."""
        adj = build_adj(n, edges)
        # Start with a random valid-list assignment
        import random
        rng = random.Random(42)
        colour = [rng.choice(lists[v]) for v in range(n)]

        def cost():
            v = 0
            for u, w in edges:
                if colour[u] == colour[w]:
                    v += 1
            return v

        best_colour = list(colour)
        best_cost = cost()
        if best_cost == 0:
            return best_colour

        t = 0
        max_iter = 200000
        while t < max_iter and time.time() < deadline:
            t += 1
            # Pick a random vertex that's in a conflict, or random vertex
            in_conflict = [v for v in range(n) if any(colour[v] == colour[u] for u in adj[v])]
            if in_conflict:
                v = rng.choice(in_conflict)
            else:
                break
            # Try changing v to a different colour in its list
            old_c = colour[v]
            for c in lists[v]:
                if c == old_c:
                    continue
                colour[v] = c
                new_cost = cost()
                if new_cost <= best_cost:
                    best_cost = new_cost
                    best_colour = list(colour)
                    if new_cost == 0:
                        return best_colour
                else:
                    colour[v] = old_c
            # If no improvement, keep the best move or revert
            colour[v] = old_c

        # Try another restart with different seed
        for seed_try in range(5):
            rng2 = random.Random(100 + seed_try)
            colour = [rng2.choice(lists[v]) for v in range(n)]
            best_colour = list(colour)
            best_cost = cost()
            if best_cost == 0:
                return best_colour
            for _ in range(max_iter // 2):
                if time.time() >= deadline:
                    break
                in_conflict = [v for v in range(n) if any(colour[v] == colour[u] for u in adj[v])]
                if not in_conflict:
                    break
                v = rng2.choice(in_conflict)
                old_c = colour[v]
                improved = False
                for c in lists[v]:
                    if c == old_c:
                        continue
                    colour[v] = c
                    new_cost = cost()
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_colour = list(colour)
                        improved = True
                        if new_cost == 0:
                            return best_colour
                    else:
                        colour[v] = old_c
                if not improved:
                    colour[v] = old_c
            if best_cost == 0:
                return best_colour

        return best_colour

    out = []
    deadline = time.time() + 110  # Safety margin before 120s

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)

        # Try backtracking first
        result = backtrack_color(n, edges, lists, deadline)

        if result is None or not is_full(result, n, edges, lists):
            # Fall back to local search
            result = local_search(n, edges, lists, deadline)

        # Final validation and type conversion
        if result is None:
            # Last resort: pick any valid list colour (may not be proper)
            result = [lists[v][0] for v in range(n)]

        # Ensure types are correct
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
