"""Strong edge colouring: multi-strategy DSATUR + induced matching peeling."""
import time
from random import Random

_T0 = time.time()
_DEADLINE = 145.0


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
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
            else:
                if (min(a, c), max(a, c)) in edge_set or \
                   (min(a, d), max(a, d)) in edge_set or \
                   (min(b, c), max(b, c)) in edge_set or \
                   (min(b, d), max(b, d)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
    return conflict


def _dsatur_color(conflict, rng=None):
    m = len(conflict)
    if m == 0:
        return []
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
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


def _dsatur_randomized(conflict, rng):
    m = len(conflict)
    if m == 0:
        return []
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
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


def _find_large_induced_matching(edges, n, conflict, remaining, rng):
    """Find a large induced matching among remaining edges."""
    m = len(edges)
    if not remaining:
        return []

    # Greedy: pick edges that don't conflict with each other
    # Sort by degree in conflict (among remaining) to pick "safe" edges first
    rem_set = set(remaining)
    # Count conflicts within remaining
    rem_deg = {}
    for i in remaining:
        cnt = 0
        for j in conflict[i]:
            if j in rem_set:
                cnt += 1
        rem_deg[i] = cnt

    # Try multiple random orderings, keep best
    best_match = []
    for trial in range(3):
        candidates = list(remaining)
        if trial == 0:
            candidates.sort(key=lambda x: rem_deg[x])
        else:
            rng.shuffle(candidates)

        chosen = []
        chosen_set = set()
        for i in candidates:
            if i not in rem_set:
                continue
            # Check if i conflicts with any already chosen
            ok = True
            for j in chosen_set:
                if j in conflict[i]:
                    ok = False
                    break
            if ok:
                chosen.append(i)
                chosen_set.add(i)
                rem_set.discard(i)

        if len(chosen) > len(best_match):
            best_match = chosen

    return best_match


def _peeling_color(edges, n, conflict, rng, time_limit):
    """Color by repeatedly finding large induced matchings."""
    m = len(edges)
    colors = [0] * m
    remaining = list(range(m))
    color_count = 0

    while remaining and time.time() < time_limit:
        match = _find_large_induced_matching(edges, n, conflict, remaining, rng)
        if not match:
            # Fallback: single edge
            match = [remaining[0]]
        for i in match:
            colors[i] = color_count
        for i in match:
            remaining.remove(i)
        color_count += 1

    return colors, color_count


def _try_reduce(conflict, colors, k, time_limit):
    """Try to reduce coloring from k to k-1 colors."""
    m = len(colors)
    last_color = k - 1
    vertices_with_last = [v for v in range(m) if colors[v] == last_color]

    if not vertices_with_last:
        return None

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
    m = len(colors)
    best = list(colors)
    best_k = k

    while best_k > 1 and time.time() < time_limit and _time_left() > 10.0:
        new_colors = _try_reduce(conflict, best, best_k, time_limit)
        if new_colors is not None:
            best = new_colors
            best_k -= 1
            continue

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

        if not improved:
            break

    return best, best_k


def _solve_instance(n, d, seed):
    edges = _build_instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []

    conflict = _build_conflict(edges, n)

    best_colors = None
    best_k = m + 1
    rng = Random(seed + 7777)

    # Strategy 1: Standard DSATUR
    colors1 = _dsatur_color(conflict)
    k1 = max(colors1) + 1 if colors1 else 0
    if k1 < best_k:
        best_colors = colors1
        best_k = k1

    # Strategy 2: Multiple randomized DSATUR
    if _time_left() > 20.0:
        for trial in range(8):
            if _time_left() < 15.0:
                break
            colors_r = _dsatur_randomized(conflict, rng)
            k_r = max(colors_r) + 1 if colors_r else 0
            if k_r < best_k:
                best_colors = colors_r
                best_k = k_r

    # Strategy 3: Peeling
    if _time_left() > 20.0:
        t_lim = min(time.time() + 8.0, _T0 + _DEADLINE - 12.0)
        colors_p, k_p = _peeling_color(edges, n, conflict, rng, t_lim)
        if k_p < best_k:
            best_colors = colors_p
            best_k = k_p

    # Local search reduction on best
    if _time_left() > 15.0 and best_colors is not None:
        t_lim = min(time.time() + 25.0, _T0 + _DEADLINE - 10.0)
        best_colors, best_k = _local_search_reduce(conflict, best_colors, best_k, t_lim)

    return best_colors


def construct():
    global _T0
    _T0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 15.0:
            # Fallback greedy
            edges = _build_instance(*spec)
            n = spec[0]
            m = len(edges)
            if m == 0:
                results.append([])
                continue
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
