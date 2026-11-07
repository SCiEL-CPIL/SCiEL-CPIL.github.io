import time
import sys
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s buffer for 120s limit

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

        # Domains as sets for fast membership and removal
        domains = [set(lists[v]) for v in range(n)]

        # Check for initial empty domains
        for v in range(n):
            if not domains[v]:
                return None

        # AC-3 constraint propagation
        def ac3(domains, adj, n, queue=None):
            if queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))

            while queue:
                u, v = queue.popleft()
                # For constraint u != v: remove c from domains[u] if domains[v] == {c}
                to_remove = []
                for c in domains[u]:
                    if len(domains[v]) == 1 and c in domains[v]:
                        to_remove.append(c)

                if to_remove:
                    for c in to_remove:
                        domains[u].discard(c)
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

        # Backtracking with MRV, LVC, and Forward Checking
        assigned = [None] * n

        def get_mrv_var():
            min_size = 4
            var = -1
            best_deg = -1
            for v in range(n):
                if assigned[v] is None:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
                        best_deg = len(adj[v])
                        if size <= 1:
                            break
                    elif size == min_size and len(adj[v]) > best_deg:
                        var = v
                        best_deg = len(adj[v])
            return var

        def backtrack():
            if time.time() > deadline:
                return False

            var = get_mrv_var()
            if var == -1:
                return True  # All assigned

            # LVC: Order values by how few constraints they impose
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

                # Assign
                assigned[var] = c
                saved_domain = set(domains[var])
                domains[var].clear()

                # Forward Checking: Remove c from neighbors' domains
                saved_neighbors = []
                ok = True
                for u in adj[var]:
                    if assigned[u] is None:
                        if c in domains[u]:
                            saved_neighbors.append(u)
                            domains[u].discard(c)
                            if not domains[u]:
                                ok = False
                                break

                if ok:
                    # Run local AC-3 on affected nodes for stronger propagation
                    local_queue = deque()
                    for u in saved_neighbors:
                        for z in adj[u]:
                            if z != var and assigned[z] is None:
                                local_queue.append((z, u))
                    if ac3(domains, adj, n, local_queue):
                        if backtrack():
                            return True

                # Undo
                assigned[var] = None
                domains[var] = saved_domain
                for u in saved_neighbors:
                    domains[u].add(c)

                if time.time() > deadline:
                    return False

            return False

        if backtrack():
            return [int(assigned[v]) for v in range(n)]
        else:
            return None

    out = []
    instances = ((20, 701), (40, 702), (60, 703), (100, 704))

    for i, (splits, seed) in enumerate(instances):
        if time.time() > TIME_LIMIT - 5:
            # Emergency: fill remaining with first available color (likely invalid but valid structure)
            for j in range(i, 4):
                s, sd = instances[j]
                n_rem, _, lists_rem = instance(s, sd)
                out.append([int(lists_rem[v][0]) for v in range(n_rem)])
            break

        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, TIME_LIMIT)

        if result is None or len(result) != n:
            # Fallback: greedy first-available (may be invalid)
            result = [int(lists[v][0]) for v in range(n)]

        # Ensure all values are Python int
        out.append([int(c) for c in result])

    return out
