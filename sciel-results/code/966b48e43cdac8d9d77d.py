"""Combined program: AC-3 + MRV/LCV backtracking with bipartite structural solver and diversified greedy."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque


def construct():
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

    def is_valid(colour, n, edges, lists):
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

    def greedy_diversified(n, adj, lists, rng):
        """Try multiple greedy orderings - fast for easy instances."""
        orders = []
        # Degree descending
        orders.append(sorted(range(n), key=lambda v: -len(adj[v])))
        # BFS order
        seen = [False] * n
        q = deque([0])
        seen[0] = True
        bfs_order = []
        while q:
            u = q.popleft()
            bfs_order.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        for v in range(n):
            if not seen[v]:
                bfs_order.append(v)
        orders.append(bfs_order)
        # Random shuffle
        rand_order = list(range(n))
        rng.shuffle(rand_order)
        orders.append(rand_order)

        for order in orders:
            colour = [-1] * n
            valid = True
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                avail = [c for c in lists[v] if c not in taken]
                if avail:
                    colour[v] = avail[0]
                else:
                    valid = False
                    break
            if valid and all(c != -1 for c in colour):
                return colour
        return None

    def ac3(domains, adj, n):
        """Arc consistency algorithm AC-3."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            u, v = queue.popleft()
            revised = False
            to_remove = []
            for c in domains[u]:
                if len(domains[v]) == 1 and list(domains[v])[0] == c:
                    to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    domains[u].discard(c)
                revised = True
                if not domains[u]:
                    return False
                for z in adj[u]:
                    if z != v:
                        queue.append((z, u))
        return True

    def ac3_propagate(domains, adj, n, changed):
        """AC-3 that starts from a set of changed nodes."""
        queue = deque()
        for v in changed:
            for u in adj[v]:
                queue.append((u, v))
        while queue:
            u, v = queue.popleft()
            revised = False
            to_remove = []
            for c in domains[u]:
                if len(domains[v]) == 1 and list(domains[v])[0] == c:
                    to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    domains[u].discard(c)
                revised = True
                if not domains[u]:
                    return False
                for z in adj[u]:
                    if z != v:
                        queue.append((z, u))
        return True

    def backtrack_ac3(n, adj, edges, lists, deadline):
        """Backtracking with AC-3 propagation and MRV+LVC heuristics."""
        domains = [set(lists[v]) for v in range(n)]
        for v in range(n):
            if not domains[v]:
                return None

        # Initial AC-3
        if not ac3(domains, adj, n):
            return None

        assigned = [-1] * n
        sys.setrecursionlimit(500000)

        def get_mrv_var():
            min_size = float('inf')
            var = -1
            max_deg = -1
            for v in range(n):
                if assigned[v] == -1:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
                        max_deg = len(adj[v])
                        if size <= 1:
                            break
                    elif size == min_size and len(adj[v]) > max_deg:
                        var = v
                        max_deg = len(adj[v])
            return var

        def solve():
            if time.time() > deadline:
                return False

            var = get_mrv_var()
            if var == -1:
                return True  # All assigned

            # LVC: order values by least constraining value
            values = list(domains[var])

            def value_score(c):
                count = 0
                for u in adj[var]:
                    if assigned[u] == -1 and c in domains[u]:
                        count += 1
                return count

            values.sort(key=value_score)

            for c in values:
                if c not in domains[var]:
                    continue

                # Assign
                assigned[var] = c
                saved_var_domain = set(domains[var])
                domains[var].clear()

                # Forward checking + AC-3 propagation
                changed = [var]
                saved_neighbors = {}
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1 and c in domains[u]:
                        saved_neighbors[u] = c
                        domains[u].discard(c)
                        if not domains[u]:
                            ok = False
                            break
                        changed.append(u)

                if ok:
                    if ac3_propagate(domains, adj, n, changed):
                        if solve():
                            return True

                # Undo
                assigned[var] = -1
                domains[var] = saved_var_domain
                for u, c2 in saved_neighbors.items():
                    domains[u].add(c2)

                if time.time() > deadline:
                    return False

            return False

        if solve():
            return [int(assigned[v]) for v in range(n)]
        return None

    def bipartite_solve(n, adj, edges, lists, deadline):
        """Exploit bipartite structure: assign one side, greedily assign the other."""
        color = [-1] * n
        parts = [[], []]
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            parts[0].append(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        parts[color[v]].append(v)
                        q.append(v)

        if len(parts[0]) <= len(parts[1]):
            small, large = parts[0], parts[1]
        else:
            small, large = parts[1], parts[0]

        small_set = set(small)
        large_set = set(large)
        large_neighbors = {v: [u for u in adj[v] if u in small_set] for v in large}
        small_neighbors = {v: [u for u in adj[v] if u in large_set] for v in small}

        colour = [-1] * n
        sys.setrecursionlimit(500000)

        def solve_small(idx):
            if time.time() > deadline:
                return False
            if idx == len(small):
                for v in large:
                    used_colors = set()
                    for u in large_neighbors[v]:
                        used_colors.add(colour[u])
                    available = [c for c in lists[v] if c not in used_colors]
                    if not available:
                        return False
                    colour[v] = available[0]
                return True

            v = small[idx]
            for c in lists[v]:
                colour[v] = c
                feasible = True
                for u in small_neighbors[v]:
                    used = set()
                    for w in large_neighbors[u]:
                        if colour[w] != -1:
                            used.add(colour[w])
                    if all(c2 in used for c2 in lists[u]):
                        feasible = False
                        break
                if feasible:
                    if solve_small(idx + 1):
                        return True
                colour[v] = -1
            return False

        if solve_small(0):
            return [int(c) for c in colour]
        return None

    def local_search(n, adj, edges, lists, deadline, rng_seed):
        """Simulated Annealing as last resort."""
        rng = Random(rng_seed)
        colour = [rng.choice(lists[v]) for v in range(n)]

        conflict_count = [0] * n
        total_conflicts = 0
        for u, v in edges:
            if colour[u] == colour[v]:
                conflict_count[u] += 1
                conflict_count[v] += 1
                total_conflicts += 1

        if total_conflicts == 0:
            return list(colour)

        best_colour = list(colour)
        best_conflicts = total_conflicts
        current_conflicts = total_conflicts
        T = 1.0
        cooling = 0.9995
        max_iter = 100000
        start_t = time.time()

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            conflicted = [v for v in range(n) if conflict_count[v] > 0]
            if conflicted:
                v = rng.choice(conflicted)
            else:
                v = rng.randrange(n)

            old_c = colour[v]
            candidates = [c for c in lists[v] if c != old_c]
            if not candidates:
                continue
            c = rng.choice(candidates)

            delta = 0
            for u in adj[v]:
                if colour[u] == old_c:
                    delta += 1
                if colour[u] == c:
                    delta -= 1

            if delta <= 0 or rng.random() < (10 ** (-current_conflicts / max(T, 0.01))):
                colour[v] = c
                for u in adj[v]:
                    if colour[u] == old_c:
                        conflict_count[u] -= 1
                        conflict_count[v] -= 1
                    if colour[u] == c:
                        conflict_count[u] += 1
                        conflict_count[v] += 1
                current_conflicts += delta
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_colour = list(colour)
                    if best_conflicts == 0:
                        return best_colour

            T *= cooling
            if T < 0.001:
                break

        return best_colour if best_conflicts == 0 else None

    def solve_instance(n, adj, edges, lists, deadline):
        """Strategy:
        1. Quick diversified greedy
        2. Bipartite structural backtracking
        3. General AC-3 backtracking with MRV/LVC
        4. Local search
        """
        rng = Random(123)

        # 1. Fast Greedy
        result = greedy_diversified(n, adj, lists, rng)
        if result is not None and is_valid(result, n, edges, lists):
            return result

        # 2. Bipartite Solver
        result = bipartite_solve(n, adj, edges, lists, deadline)
        if result is not None and is_valid(result, n, edges, lists):
            return result

        # 3. AC-3 Backtracking
        result = backtrack_ac3(n, adj, edges, lists, deadline)
        if result is not None and is_valid(result, n, edges, lists):
            return result

        # 4. Local Search
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            result = local_search(n, adj, edges, lists, deadline, seed)
            if result is not None and is_valid(result, n, edges, lists):
                return result

        # Fallback
        return [lists[v][0] for v in range(n)]

    out = []
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        result = solve_instance(n, adj, edges, lists, deadline)

        # Ensure types are strictly Python int
        result = [int(c) for c in result]
        out.append(result)

    return out
# EVOLVE-BLOCK-END
