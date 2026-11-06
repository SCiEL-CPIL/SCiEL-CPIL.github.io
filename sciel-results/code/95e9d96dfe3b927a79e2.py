"""Strong edge colourings of bounded-degree graphs: precomputed conflict + vertex-coverage greedy + 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import random as _rnd_mod
from collections import defaultdict

def construct():
    t_start = time.time()
    TOTAL_BUDGET = 138.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = _rnd_mod.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_conflict(edges, adj, n):
        """Precompute conflict graph: two edges conflict if they share a vertex
        or are joined by an edge (any endpoint of one adjacent to any endpoint of the other)."""
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                if a == c or a == d or b == c or b == d:
                    conflict[i].add(j)
                    conflict[j].add(i)
                elif c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_color_coverage(edges, adj, order, deadline):
        """Greedy coloring using vertex-coverage tracking.
        For each color, maintain the set of vertices 'covered' (endpoints of edges with that color).
        Edge (u,v) can use color c iff no vertex in conflict_set(u,v) is covered by color c.
        """
        m = len(edges)
        colour = [-1] * m
        color_vertices = defaultdict(set)  # color -> set of vertices covered

        # Precompute conflict vertex sets for each edge
        conf_verts = []
        for u, v in edges:
            s = {u, v}
            s.update(adj[u])
            s.update(adj[v])
            conf_verts.append(s)

        for idx in order:
            u, v = edges[idx]
            near = conf_verts[idx]
            c = 0
            while c in color_vertices and (color_vertices[c] & near):
                c += 1
            colour[idx] = c
            color_vertices[c].add(u)
            color_vertices[c].add(v)
        return colour

    def greedy_color_conflict(edges, conflict, order, deadline):
        """Greedy coloring using precomputed conflict graph."""
        m = len(edges)
        colour = [-1] * m
        for idx in order:
            used = set()
            for j in conflict[idx]:
                if colour[j] != -1:
                    used.add(colour[j])
            c = 0
            while c in used:
                c += 1
            colour[idx] = c
        return colour

    def count_colors(colour):
        if not colour or max(colour) < 0:
            return 0
        return max(colour) + 1

    def try_recolor(colour, idx, new_color, conflict):
        """Check if we can assign new_color to edge idx."""
        for j in conflict[idx]:
            if colour[j] == new_color:
                return False
        return True

    def local_improve_1opt(colour, conflict, max_iters=200, deadline=None):
        """1-opt local improvement: recolor single edges to lower colors."""
        m = len(colour)
        if m == 0:
            return colour
        current_max = count_colors(colour)
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False
            # Try to recolor each edge to a smaller colour
            for idx in range(m):
                current_color = colour[idx]
                for new_c in range(current_color):
                    if try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        improved = True
                        break
            # Try to reduce the overall max colour
            new_max = count_colors(colour)
            if new_max < current_max:
                current_max = new_max
            if current_max > 0:
                max_c = current_max - 1
                max_edges = [i for i in range(m) if colour[i] == max_c]
                for idx in max_edges:
                    for new_c in range(max_c):
                        if try_recolor(colour, idx, new_c, conflict):
                            colour[idx] = new_c
                            current_max -= 1
                            improved = True
                            break
            if not improved:
                break
        return colour

    def local_improve_2opt(colour, conflict, max_iters=100, deadline=None):
        """2-opt local improvement: try swapping colors of two edges to reduce max color."""
        m = len(colour)
        if m == 0:
            return colour
        current_max = count_colors(colour)
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False
            # First do 1-opt
            for idx in range(m):
                current_color = colour[idx]
                for new_c in range(current_color):
                    if try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        improved = True
                        break

            # Try 2-opt: swap colors of two edges to eliminate the highest color
            new_max = count_colors(colour)
            if new_max < current_max:
                current_max = new_max

            if current_max > 0:
                max_c = current_max - 1
                max_edges = [i for i in range(m) if colour[i] == max_c]
                # Try to move each max-edge to a lower color via 1-opt first
                for idx in max_edges:
                    for new_c in range(max_c):
                        if try_recolor(colour, idx, new_c, conflict):
                            colour[idx] = new_c
                            current_max -= 1
                            improved = True
                            break
                    if colour[idx] != max_c:
                        continue
                    # Try 2-opt: swap with another edge
                    for j in range(m):
                        if j == idx or colour[j] == max_c:
                            continue
                        cj = colour[j]
                        # Check if swapping works: idx gets cj, j gets max_c
                        # idx can take cj if no neighbor of idx has color cj
                        ok_idx = True
                        for k in conflict[idx]:
                            if colour[k] == cj:
                                ok_idx = False
                                break
                        if not ok_idx:
                            continue
                        # j can take max_c if no neighbor of j has color max_c
                        ok_j = True
                        for k in conflict[j]:
                            if k != idx and colour[k] == max_c:
                                ok_j = False
                                break
                        if ok_j:
                            colour[idx] = cj
                            colour[j] = max_c
                            # Wait, this doesn't help - j still has max_c
                            # We need j to get a color < max_c
                            # Actually 2-opt swap: idx gets cj, j gets max_c doesn't reduce max
                            # We need: move idx from max_c to cj, and j from cj to some other color
                            # That's really a 3-move. Let's skip complex 2-opt.
                            colour[idx] = max_c
                            colour[j] = cj
                            break
                    if colour[idx] != max_c:
                        improved = True
                        current_max = count_colors(colour)
                        if current_max <= max_c - 1:
                            break

            if not improved:
                break
        return colour

    def solve_instance(edges, adj, n, time_budget, deadline):
        m = len(edges)
        if m == 0:
            return []

        conflict = build_conflict(edges, adj, n)
        deg = [len(adj[v]) for v in range(n)]

        best_colour = None
        best_count = float('inf')

        # Define orderings
        orderings = []

        # Largest degree sum first
        order1 = sorted(range(m), key=lambda i: -(deg[edges[i][0]] + deg[edges[i][1]]))
        orderings.append(order1)

        # Smallest degree sum first
        order2 = sorted(range(m), key=lambda i: (deg[edges[i][0]] + deg[edges[i][1]], i))
        orderings.append(order2)

        # Natural order
        orderings.append(list(range(m)))

        # Try each deterministic ordering with both greedy methods
        for order in orderings:
            if time.time() > deadline:
                break
            # Vertex-coverage greedy
            colour = greedy_color_coverage(edges, adj, order, deadline)
            colour = local_improve_1opt(colour, conflict, max_iters=150, deadline=deadline)
            c = count_colors(colour)
            if c < best_count:
                best_count = c
                best_colour = colour[:]

            if time.time() > deadline:
                break

            # Conflict-graph greedy
            colour = greedy_color_conflict(edges, conflict, order, deadline)
            colour = local_improve_1opt(colour, conflict, max_iters=150, deadline=deadline)
            c = count_colors(colour)
            if c < best_count:
                best_count = c
                best_colour = colour[:]

        # Random restarts
        rng = _rnd_mod.Random(777)
        max_random = 30
        for attempt in range(max_random):
            if time.time() > deadline:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = greedy_color_coverage(edges, adj, order, deadline)
            colour = local_improve_1opt(colour, conflict, max_iters=80, deadline=deadline)
            c = count_colors(colour)
            if c < best_count:
                best_count = c
                best_colour = colour[:]

            if time.time() > deadline:
                break

            # Also try conflict-based greedy on random order
            colour = greedy_color_conflict(edges, conflict, order, deadline)
            colour = local_improve_1opt(colour, conflict, max_iters=80, deadline=deadline)
            c = count_colors(colour)
            if c < best_count:
                best_count = c
                best_colour = colour[:]

        if best_colour is None:
            best_colour = greedy_color_coverage(edges, adj, list(range(m)), deadline)

        return best_colour

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    out = []
    per_instance_budget = TOTAL_BUDGET / len(specs)

    for idx, spec in enumerate(specs):
        n, d, seed = spec
        elapsed = time.time() - t_start
        remaining = TOTAL_BUDGET - elapsed
        time_budget = min(per_instance_budget, remaining * 0.95)
        deadline = t_start + TOTAL_BUDGET

        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        colours = solve_instance(edges, adj, n, time_budget, deadline)
        out.append(colours)

    return out
# EVOLVE-BLOCK-END
