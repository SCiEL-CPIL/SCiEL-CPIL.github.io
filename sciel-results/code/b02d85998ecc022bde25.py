import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_incompatibility(edges, adj):
        m = len(edges)
        incompatible = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        return incompatible

    def greedy_coloring(incompatible, m, order=None):
        """Greedy coloring with a given vertex order."""
        if order is None:
            order = list(range(m))
        colors = [-1] * m
        for v in order:
            used = set()
            for u in incompatible[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return colors

    def dsatur_coloring(incompatible, m):
        """DSATUR: color vertex with highest saturation first, tie-break by degree."""
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        degree = [len(incompatible[i]) for i in range(m)]
        colored = [False] * m

        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(m):
                if not colored[v]:
                    sat = len(saturation[v])
                    if sat > best_sat or (sat == best_sat and degree[v] > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]

            if best_v == -1:
                break

            used = set(saturation[best_v])
            c = 0
            while c in used:
                c += 1
            colors[best_v] = c
            colored[best_v] = True
            for u in incompatible[best_v]:
                if not colored[u]:
                    saturation[u].add(c)

        return colors

    def count_colors(colors):
        return max(colors) + 1 if colors else 0

    def local_search(incompatible, colors, m, time_limit):
        """Local search: try moves and swaps to reduce color count."""
        best_colors = colors[:]
        best_k = count_colors(best_colors)
        rng = random.Random(42)
        start = time.time()

        while time.time() - start < time_limit:
            improved = False

            # Try moving a vertex to a lower color
            for v in rng.sample(range(m), m):
                cv = best_colors[v]
                if cv == 0:
                    continue
                for c in range(cv):
                    conflict = False
                    for u in incompatible[v]:
                        if best_colors[u] == c:
                            conflict = True
                            break
                    if not conflict:
                        best_colors[v] = c
                        improved = True
                        break
                if improved:
                    break

            if improved:
                new_k = count_colors(best_colors)
                if new_k < best_k:
                    best_k = new_k
                continue

            # Try swapping two vertices
            swapped = False
            for _ in range(50):
                v1, v2 = rng.sample(range(m), 2)
                c1, c2 = best_colors[v1], best_colors[v2]
                if c1 == c2:
                    continue
                ok1 = True
                for u in incompatible[v1]:
                    if u == v2:
                        continue
                    if best_colors[u] == c2:
                        ok1 = False
                        break
                if not ok1:
                    continue
                ok2 = True
                for u in incompatible[v2]:
                    if u == v1:
                        continue
                    if best_colors[u] == c1:
                        ok2 = False
                        break
                if not ok2:
                    continue
                best_colors[v1] = c2
                best_colors[v2] = c1
                swapped = True
                break

            if swapped:
                new_k = count_colors(best_colors)
                if new_k < best_k:
                    best_k = new_k

            if not improved and not swapped:
                if rng.random() < 0.1:
                    v = rng.randrange(m)
                    new_c = rng.randint(0, best_k)
                    best_colors[v] = new_c
                else:
                    break

        return best_colors

    def max_clique_branch_and_bound(adjacency, n, time_budget):
        """
        Maximum clique branch-and-bound on a compatibility graph.
        adjacency: list of sets, where adjacency[i] contains all j compatible with i.
        Returns a list of vertex indices forming a maximum clique found within time_budget.
        """
        best_clique = []
        best_size = 0
        start = time.time()
        calls = [0]

        # Sort vertices by degree in descending order for better pruning
        order = sorted(range(n), key=lambda x: -len(adjacency[x]))
        pos_in_order = [0] * n
        for idx, v in enumerate(order):
            pos_in_order[v] = idx

        # Rebuild adjacency in terms of order position for efficiency
        # Actually, let's just use the original indices but order the candidate set.

        def expand(clique, candidates):
            nonlocal best_clique, best_size
            calls[0] += 1
            if time.time() - start > time_budget:
                return

            if len(candidates) == 0:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return

            # Pruning: if len(clique) + len(candidates) <= best_size, prune
            if len(clique) + len(candidates) <= best_size:
                return

            # Pick a vertex from candidates with highest degree in candidates
            # To speed up, we can just pick the last one or one with max deg
            # Let's pick the vertex with the most neighbors in candidates
            max_deg = -1
            chosen = -1
            for v in candidates:
                deg = sum(1 for u in candidates if u in adjacency[v])
                if deg > max_deg:
                    max_deg = deg
                    chosen = v

            if chosen == -1:
                return

            # Branch: include chosen
            new_clique = clique + [chosen]
            new_candidates = [u for u in candidates if u in adjacency[chosen] and u != chosen]
            expand(new_clique, new_candidates)

            # Branch: exclude chosen
            candidates.remove(chosen)
            expand(clique, candidates)
            candidates.append(chosen)  # Restore

        # Initial candidate set is all vertices
        candidates = list(range(n))
        expand([], candidates)

        return best_clique

    def greedy_clique_cover(incompatible, m, time_budget):
        """
        Greedily find cliques in the compatibility graph (independent sets in incompatibility).
        Each clique is a color class.
        """
        # Compatibility graph: edge between i, j if i, j are NOT incompatible
        # adjacency_compat[i] = set of j such that j not in incompatible[i] and j != i
        compat_adj = [set() for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i != j and j not in incompatible[i]:
                    compat_adj[i].add(j)

        remaining = set(range(m))
        color_classes = []
        start = time.time()

        while remaining:
            if time.time() - start > time_budget:
                break

            # Find a maximal clique in the compatibility graph restricted to remaining
            # Use a simple greedy approach for speed, or small B&B
            # Greedy: pick vertex with max degree in remaining, add it, remove incompatible
            # But we want a large clique.
            # Let's use a greedy heuristic: pick vertex with max degree in the subgraph induced by remaining
            if not remaining:
                break

            # Order remaining by degree in compat graph (within remaining)
            degs = {}
            for v in remaining:
                degs[v] = len(compat_adj[v] & remaining)

            # Start with a random vertex or the one with max degree
            # To get a good clique, we can use a simple greedy:
            # 1. Pick a vertex v with max degree in remaining
            # 2. Add v to clique
            # 3. Remove all vertices not compatible with v from remaining
            # 4. Repeat until no vertices left or budget exceeded

            clique = []
            current_remaining = set(remaining)

            while current_remaining:
                if time.time() - start > time_budget:
                    break
                # Pick vertex with max degree in current_remaining
                best_v = -1
                best_deg = -1
                for v in current_remaining:
                    deg = len(compat_adj[v] & current_remaining)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

                if best_v == -1:
                    break

                clique.append(best_v)
                # Keep only vertices compatible with best_v
                current_remaining = {v for v in current_remaining if best_v in compat_adj[v]}
                current_remaining.discard(best_v)

            if clique:
                color_classes.append(clique)
                for v in clique:
                    remaining.discard(v)

        # Assign colors
        colors = [-1] * m
        for c_idx, clique in enumerate(color_classes):
            for v in clique:
                colors[v] = c_idx

        # Handle any remaining vertices (if budget ran out)
        if remaining:
            # Greedy color the rest
            for v in remaining:
                used = set()
                for u in incompatible[v]:
                    if colors[u] != -1:
                        used.add(colors[u])
                c = 0
                while c in used:
                    c += 1
                colors[v] = c

        return colors

    def solve_instance(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return []

        incompatible = build_incompatibility(edges, adj)

        # 1. Get an initial coloring via DSATUR
        colors = dsatur_coloring(incompatible, m)
        k = count_colors(colors)

        # 2. Try greedy clique cover to see if it's better
        # Allocate half the time to this
        clique_colors = greedy_clique_cover(incompatible, m, time_budget * 0.4)
        k_clique = count_colors(clique_colors)

        if k_clique < k:
            colors = clique_colors
            k = k_clique

        # 3. Local search to improve
        colors = local_search(incompatible, colors, m, time_budget * 0.6)

        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining_time = deadline - time.time()
        if remaining_time <= 5:
            # Fallback: just use DSATUR
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            incompatible = build_incompatibility(edges, adj)
            colors = dsatur_coloring(incompatible, len(edges))
            out.append(colors)
            continue

        # Allocate time based on instance size
        # Smaller instances get less time, larger get more
        # But we need to ensure all are solved.
        # Let's give proportional time based on n^2
        n = spec[0]
        # Weights: 40^2=1600, 60^2=3600, 80^2=6400. Total=11600
        # But the first instance is quick.
        # Let's just split remaining time 30%, 30%, 40%
        if len(out) == 0:
            time_budget = remaining_time * 0.3
        elif len(out) == 1:
            time_budget = remaining_time * 0.3
        else:
            time_budget = remaining_time * 0.4

        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        colors = solve_instance(edges, adj, time_budget)
        out.append(colors)

    return out
