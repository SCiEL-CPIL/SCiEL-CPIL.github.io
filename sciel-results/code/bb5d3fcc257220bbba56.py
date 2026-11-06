import time
import math
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_left():
        return _DEADLINE - (time.time() - _T0)

    def instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph
        # Conflict if share vertex or connected by edge
        adj_orig = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share vertex?
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    # Joined by an edge?
                    if (min(a, c), max(a, c)) in edge_set or \
                       (min(a, dd), max(a, dd)) in edge_set or \
                       (min(b, c), max(b, c)) in edge_set or \
                       (min(b, dd), max(b, dd)) in edge_set:
                        neigh[i].add(j)
                        neigh[j].add(i)

        # Convert to list for faster iteration
        neigh_list = [list(s) for s in neigh]

        def find_max_is(active_set, rng):
            """
            Find a large independent set in the subgraph induced by active_set.
            Uses a greedy heuristic with randomization.
            """
            if not active_set:
                return set()

            active = set(active_set)
            is_set = set()

            while active:
                # Find vertex with min degree in active subgraph
                min_deg = float('inf')
                candidates = []
                for v in active:
                    deg = 0
                    for u in neigh_list[v]:
                        if u in active:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)

                # Random selection among min-degree candidates to break ties randomly
                v = rng.choice(candidates)

                is_set.add(v)

                # Remove v and its neighbors from active
                active.remove(v)
                for u in neigh_list[v]:
                    if u in active:
                        active.remove(u)

            return is_set

        def solve_with_peeling(rng, time_limit):
            active = set(range(m))
            colors = [-1] * m
            current_color = 0

            while active:
                if time.time() > time_limit:
                    break

                best_is = set()
                remaining_time = time_limit - time.time()
                if remaining_time < 0:
                    break

                # Heuristic: more restarts when we have more time
                # But we must ensure we finish.
                # Let's do a fixed number of restarts, say 50-200, depending on m.
                num_restarts = 100
                if len(active) < 20:
                    num_restarts = 500
                elif len(active) < 50:
                    num_restarts = 200
                else:
                    num_restarts = 100

                for _ in range(num_restarts):
                    if time.time() > time_limit - 0.1:
                        break
                    is_set = find_max_is(active, rng)
                    if len(is_set) > len(best_is):
                        best_is = is_set

                if not best_is:
                    best_is = {next(iter(active))}

                for v in best_is:
                    colors[v] = current_color

                active.difference_update(best_is)
                current_color += 1

            if -1 in colors:
                # Timeout or failure, assign remaining arbitrarily
                next_c = current_color
                for i in range(m):
                    if colors[i] == -1:
                        colors[i] = next_c
                        next_c += 1

            return colors

        def dsatur_coloring(conflict, rng=None):
            m_local = len(conflict)
            if m_local == 0:
                return []
            colors = [-1] * m_local
            sat = [set() for _ in range(m_local)]
            uncolored = list(range(m_local))
            color_count = 0

            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    s = len(sat[v])
                    dg = len(conflict[v])
                    if s > best_sat or (s == best_sat and dg > best_deg):
                        best_v = v
                        best_sat = s
                        best_deg = dg

                c = 0
                while c in sat[best_v]:
                    c += 1
                colors[best_v] = c
                if c == color_count:
                    color_count += 1

                for nb in conflict[best_v]:
                    if colors[nb] == -1:
                        sat[nb].add(c)

                uncolored.remove(best_v)

            return colors

        def dsatur_randomized(conflict, rng):
            m_local = len(conflict)
            if m_local == 0:
                return []
            colors = [-1] * m_local
            sat = [set() for _ in range(m_local)]
            uncolored = list(range(m_local))
            color_count = 0

            while uncolored:
                # Find max saturation
                max_sat = max(len(sat[v]) for v in uncolored)
                candidates = [v for v in uncolored if len(sat[v]) == max_sat]
                if len(candidates) > 1:
                    # Random tie-break among high-degree candidates
                    best_deg = max(len(conflict[v]) for v in candidates)
                    top = [v for v in candidates if len(conflict[v]) == best_deg]
                    best_v = rng.choice(top)
                else:
                    best_v = candidates[0]

                c = 0
                while c in sat[best_v]:
                    c += 1
                colors[best_v] = c
                if c == color_count:
                    color_count += 1

                for nb in conflict[best_v]:
                    if colors[nb] == -1:
                        sat[nb].add(c)

                uncolored.remove(best_v)

            return colors

        # Initialize best
        best_coloring = None
        best_num_colors = float('inf')

        # Strategy 1: Peeling with random restarts
        rng = Random(seed * 7919 + 42)
        while time.time() < deadline - 0.5:
            attempt_rng = Random(rng.random())
            coloring = solve_with_peeling(attempt_rng, deadline - 0.5)
            num_colors = max(coloring) + 1 if coloring else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]

            # Check if we hit the theoretical lower bound
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break

        # Strategy 2: DSATUR (deterministic and randomized)
        if time.time() < deadline - 0.5:
            # Deterministic DSATUR
            coloring = dsatur_coloring(neigh_list)
            num_colors = max(coloring) + 1 if coloring else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]

        # Randomized DSATUR
        rng_dsatur = Random(seed * 104729 + 13)
        while time.time() < deadline - 0.5:
            coloring = dsatur_randomized(neigh_list, rng_dsatur)
            num_colors = max(coloring) + 1 if coloring else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]

            # Check if we hit the theoretical lower bound
            en_bound = int(5 * d * d / 4)
            if best_num_colors <= en_bound:
                break

        if best_coloring is None:
            # Fallback: simple greedy
            active = set(range(m))
            colors = [-1] * m
            c = 0
            while active:
                v = next(iter(active))
                colors[v] = c
                active.remove(v)
                for u in neigh_list[v]:
                    if u in active:
                        active.remove(u)
                c += 1
            best_coloring = colors

        return compress(best_coloring)

    try:
        start = time.time()
        total_budget = 140.0
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
