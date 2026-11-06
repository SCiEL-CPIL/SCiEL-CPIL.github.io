"""Strong edge colourings of bounded-degree graphs:
vertex-coverage greedy (fast set-based conflict) + 1-opt + 2-opt local search."""
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
        """Conflict graph: two edges conflict if they share a vertex
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

    def build_conf_sets(edges, adj):
        """conf_sets[i] = set of vertices that a same-colour edge adjacent to edge i would
        be forced to use (endpoints of i plus all their neighbours)."""
        return [({u, v} | adj[u] | adj[v]) for u, v in edges]

    def greedy_color_coverage(edges, conf_sets, order):
        """Vertex-coverage greedy: edge (u,v) takes colour c iff no vertex in its
        conflict set is already covered by colour c."""
        m = len(edges)
        colour = [-1] * m
        color_vertices = {}
        for idx in order:
            u, v = edges[idx]
            near = conf_sets[idx]
            c = 0
            while c in color_vertices and (color_vertices[c] & near):
                c += 1
            colour[idx] = c
            s = color_vertices.get(c)
            if s is None:
                color_vertices[c] = {u, v}
            else:
                s.add(u)
                s.add(v)
        return colour

    def count_colors(colour):
        if not colour or max(colour) < 0:
            return 0
        return max(colour) + 1

    def try_recolor(colour, idx, new_color, conflict):
        for j in conflict[idx]:
            if colour[j] == new_color:
                return False
        return True

    def local_improve(colour, conflict, max_iters=100, deadline=None):
        """1-opt (recolor to a lower colour) + 2-opt (move the highest-colour edge into an
        existing colour by swapping one conflicting edge into the vacated colour)."""
        m = len(colour)
        if m == 0:
            return colour
        cur_max = count_colors(colour)
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False

            # 1-opt: recolor each edge to the smallest available colour
            for idx in range(m):
                cc = colour[idx]
                for new_c in range(cc):
                    if try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        improved = True
                        break

            new_max = count_colors(colour)
            if new_max < cur_max:
                cur_max = new_max
                improved = True

            # 2-opt: eliminate the highest colour
            if cur_max > 0:
                max_c = cur_max - 1
                max_edges = [i for i in range(m) if colour[i] == max_c]
                for idx in max_edges:
                    if colour[idx] != max_c:
                        continue
                    # direct 1-opt move of the max edge
                    moved = False
                    for new_c in range(max_c):
                        if try_recolor(colour, idx, new_c, conflict):
                            colour[idx] = new_c
                            moved = True
                            break
                    if moved:
                        improved = True
                        cur_max = count_colors(colour)
                        continue
                    # 2-opt: idx -> new_c, and one conflicting edge j -> max_c
                    for new_c in range(max_c):
                        blockers = [j for j in conflict[idx] if colour[j] == new_c]
                        if not blockers:
                            continue
                        for j in blockers:
                            ok = True
                            for k in conflict[j]:
                                if k != idx and colour[k] == max_c:
                                    ok = False
                                    break
                            if ok:
                                colour[idx] = new_c
                                colour[j] = max_c
                                improved = True
                                cur_max = count_colors(colour)
                                break
                        if colour[idx] != max_c:
                            break
                    if cur_max <= max_c - 1:
                        break

            if not improved:
                break
        return colour

    def solve_instance(edges, adj, n, deadline):
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict(edges, adj, n)
        conf_sets = build_conf_sets(edges, adj)
        deg = [len(adj[v]) for v in range(n)]

        best_colour = None
        best_count = float('inf')

        orderings = [
            sorted(range(m), key=lambda i: -(deg[edges[i][0]] + deg[edges[i][1]])),
            sorted(range(m), key=lambda i: (deg[edges[i][0]] + deg[edges[i][1]], i)),
            list(range(m)),
        ]

        def consider(colour):
            nonlocal best_colour, best_count
            c = count_colors(colour)
            if c < best_count:
                best_count = c
                best_colour = colour[:]

        for order in orderings:
            if time.time() > deadline:
                break
            colour = greedy_color_coverage(edges, conf_sets, order)
            colour = local_improve(colour, conflict, max_iters=120, deadline=deadline)
            consider(colour)

        rng = _rnd_mod.Random(777)
        for _ in range(40):
            if time.time() > deadline:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = greedy_color_coverage(edges, conf_sets, order)
            colour = local_improve(colour, conflict, max_iters=60, deadline=deadline)
            consider(colour)

        if best_colour is None:
            best_colour = greedy_color_coverage(edges, conf_sets, list(range(m)))
        return best_colour

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    out = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        out.append(solve_instance(edges, adj, n, t_start + TOTAL_BUDGET))

    return out
# EVOLVE-BLOCK-END
