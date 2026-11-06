"""Strong edge colouring: DSATUR + conflict-graph reduction via backtracking."""
import time
from random import Random

_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150s wall-clock kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_instance(n, d, seed):
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


def _build_conflict(edges, n):
    """Build conflict graph for strong edge coloring.
    Two edges conflict if they share a vertex OR are joined by an edge."""
    m = len(edges)
    edge_set = set(edges)
    orig_adj = [set() for _ in range(n)]
    for u, v in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)

    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            # Share a vertex?
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
            else:
                # Joined by an edge?
                if (min(a, c), max(a, c)) in edge_set or \
                   (min(a, d), max(a, d)) in edge_set or \
                   (min(b, c), max(b, c)) in edge_set or \
                   (min(b, d), max(b, d)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
    return conflict


def _dsatur_color(conflict):
    """DSATUR greedy coloring of the conflict graph."""
    m = len(conflict)
    if m == 0:
        return []
    colors = [-1] * m
    degrees = [len(conflict[i]) for i in range(m)]
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
    color_count = 0

    while uncolored:
        # Pick vertex with highest saturation, tie-break by degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            s = len(sat[v])
            dg = degrees[v]
            if s > best_sat or (s == best_sat and dg > best_deg):
                best_v = v
                best_sat = s
                best_deg = dg

        # Assign lowest available color
        c = 0
        while c in sat[best_v]:
            c += 1
        colors[best_v] = c
        if c == color_count:
            color_count += 1

        # Update saturation of neighbors
        for nb in conflict[best_v]:
            if colors[nb] == -1:
                sat[nb].add(c)

        uncolored.remove(best_v)

    return colors


def _try_reduce(conflict, colors, k, time_limit):
    """Try to reduce coloring from k to k-1 colors via backtracking on last color class."""
    m = len(colors)
    last_color = k - 1
    vertices_with_last = [v for v in range(m) if colors[v] == last_color]

    if not vertices_with_last:
        return None

    # Order by fewest options first (most constrained)
    def options_count(v):
        cnt = 0
        for c in range(k - 1):
            ok = True
            for nb in conflict[v]:
                if colors[nb] == c:
                    ok = False
                    break
            if ok:
                cnt += 1
        return cnt

    vertices_with_last.sort(key=options_count)

    new_colors = colors[:]

    def backtrack(idx):
        if time.time() > time_limit:
            return False
        if idx == len(vertices_with_last):
            return True
        v = vertices_with_last[idx]
        # Try colors 0 to k-2
        for c in range(k - 1):
            ok = True
            for nb in conflict[v]:
                if new_colors[nb] == c:
                    ok = False
                    break
            if ok:
                new_colors[v] = c
                if backtrack(idx + 1):
                    return True
                new_colors[v] = last_color
        return False

    if backtrack(0):
        return new_colors
    return None


def _local_search_reduce(conflict, colors, k, time_limit):
    """Iteratively try to reduce colors by reassigning vertices in the last color class."""
    m = len(colors)
    best = list(colors)
    best_k = k
    t_start = time.time()

    while best_k > 1 and time.time() < time_limit and _time_left() > 10.0:
        # Try backtracking reduction
        new_colors = _try_reduce(conflict, best, best_k, time_limit)
        if new_colors is not None:
            best = new_colors
            best_k -= 1
            t_start = time.time()  # reset timer for next attempt
            continue

        # Try local search: move one vertex from last color to another
        last_color = best_k - 1
        last_class = [v for v in range(m) if best[v] == last_color]
        improved = False
        for v in last_class:
            for c in range(last_color):
                ok = True
                for nb in conflict[v]:
                    if best[nb] == c:
                        ok = False
                        break
                if ok:
                    best[v] = c
                    improved = True
                    break
            if improved:
                break
            if time.time() - t_start > 2.0:
                break

        if not improved:
            break

    return best


def _solve_instance(n, d, seed):
    edges = _build_instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []

    conflict = _build_conflict(edges, n)

    # Initial DSATUR coloring
    colors = _dsatur_color(conflict)
    k = max(colors) + 1 if colors else 0

    # Try to reduce colors
    if _time_left() > 15.0:
        time_limit = min(time.time() + 20.0, _T0 + _DEADLINE - 10.0)
        colors = _local_search_reduce(conflict, colors, k, time_limit)

    return colors


def construct():
    global _T0
    _T0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 15.0:
            # Not enough time; use fast greedy
            edges = _build_instance(*spec)
            n = spec[0]
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            # Simple greedy: assign each edge the smallest color not used by conflicting edges
            edge_set = set(edges)
            colors = [0] * m
            for i in range(m):
                a, b = edges[i]
                used = set()
                for j in range(i):
                    c, d = edges[j]
                    if a == c or a == d or b == c or b == d:
                        used.add(colors[j])
                    elif (min(a, c), max(a, c)) in edge_set or \
                         (min(a, d), max(a, d)) in edge_set or \
                         (min(b, c), max(b, c)) in edge_set or \
                         (min(b, d), max(b, d)) in edge_set:
                        used.add(colors[j])
                colors[i] = 0
                while colors[i] in used:
                    colors[i] += 1
            results.append(colors)
            continue

        colors = _solve_instance(*spec)
        results.append(colors)

    return results
