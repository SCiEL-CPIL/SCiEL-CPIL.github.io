"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

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
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(colour, adj, lists):
        for v, c in enumerate(colour):
            if c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def greedy_fallback(n, adj, lists, seed=0):
        """A deterministic-ish greedy that always returns a full list of ints."""
        rng = Random(seed)
        colour = [None] * n
        # order by degree descending, break ties by index
        order = sorted(range(n), key=lambda u: (-len(adj[u]), u))
        for v in order:
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                # pick randomly among free to diversify
                colour[v] = rng.choice(free)
            else:
                # must pick something; pick first in list (will be invalid but valid structure)
                colour[v] = lists[v][0]
        # ensure all are Python ints
        return [int(c) for c in colour]

    def solve_csp(n, adj, lists, time_limit):
        """
        Backtracking with MRV heuristic, forward checking, and degree ordering.
        Returns a valid colouring or None if not found within time_limit.
        """
        start = time.time()
        colour = [None] * n
        # domains as lists
        domains = [list(l) for l in lists]
        # Precompute degree for tie-breaking
        degrees = [len(adj[v]) for v in range(n)]
        
        def mrv_vertex():
            best_v = -1
            best_size = n + 1
            for v in range(n):
                if colour[v] is None:
                    # count available
                    avail = len([c for c in domains[v] if all(c != colour[u] for u in adj[v] if colour[u] is not None)])
                    if avail < best_size or (avail == best_size and best_v != -1 and degrees[v] > degrees[best_v]):
                        best_size = avail
                        best_v = v
                        if avail == 0:
                            break
            return best_v

        def backtrack():
            if time.time() - start > time_limit:
                return False
            v = mrv_vertex()
            if v == -1:
                return True
            # Try each colour in domain[v], ordered by least constraining value
            # (fewest options removed from neighbors)
            cands = []
            for c in domains[v]:
                conflict = False
                for u in adj[v]:
                    if colour[u] is not None and colour[u] == c:
                        conflict = True
                        break
                if not conflict:
                    # Count how many options this removes from unassigned neighbors
                    removals = 0
                    for u in adj[v]:
                        if colour[u] is None and c in domains[u]:
                            removals += 1
                    cands.append((removals, c))
            cands.sort()
            for _, c in cands:
                colour[v] = c
                # Forward checking: verify no neighbor has empty domain
                fc_ok = True
                for u in adj[v]:
                    if colour[u] is None:
                        if all(c2 == c for c2 in domains[u]):
                            fc_ok = False
                            break
                if fc_ok:
                    if backtrack():
                        return True
                colour[v] = None
            return False

        if backtrack():
            return [int(c) for c in colour]
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        # Try CSP solver with a generous time limit
        remaining = deadline - time.time()
        per_instance_time = max(1.0, remaining / (4 - len(out)))
        # Use a portion for CSP, keep some for fallback
        csp_time = min(per_instance_time * 0.8, 30.0)
        
        colour = solve_csp(n, adj, lists, csp_time)
        
        if colour is None or not is_valid(colour, adj, lists):
            # Fallback: greedy with a few random restarts
            best_colour = None
            best_score = -1
            num_restarts = min(50, int(max(1, (deadline - time.time()) / 0.1)))
            for r in range(num_restarts):
                if time.time() > deadline - 5.0:
                    break
                c = greedy_fallback(n, adj, lists, seed=seed + r * 1000)
                # count valid edges
                valid_edges = 0
                total_edges = len(edges)
                for u, v in edges:
                    if c[u] != c[v]:
                        valid_edges += 1
                # also check list membership
                list_ok = all(c[v] in lists[v] for v in range(n))
                score = valid_edges if list_ok else -1
                if score > best_score:
                    best_score = score
                    best_colour = c
            if best_colour is not None:
                colour = best_colour
            else:
                colour = greedy_fallback(n, adj, lists, seed=seed)
        
        out.append(colour)
        
        # Check deadline
        if time.time() > deadline - 2.0:
            # Fill remaining with quick greedy
            for splits2, seed2 in ((20, 701), (40, 702), (60, 703), (100, 704))[len(out):]:
                n2, edges2, lists2 = instance(splits2, seed2)
                adj2 = build_adj(n2, edges2)
                out.append(greedy_fallback(n2, adj2, lists2, seed=seed2))
            break

    # Ensure all are lists of Python ints
    final_out = []
    for colour in out:
        final_out.append([int(c) for c in colour])
    return final_out
# EVOLVE-BLOCK-END
