"""Strong edge colouring: B&B MIS peeling + multi-restart + local search."""
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
    conflict = [0] * m
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
            else:
                if (min(a, c), max(a, c)) in edge_set or \
                   (min(a, d), max(a, d)) in edge_set or \
                   (min(b, c), max(b, c)) in edge_set or \
                   (min(b, d), max(b, d)) in edge_set:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
    return conflict


def _coloring_bound(conflict, candidates):
    """Upper bound on MIS size using greedy coloring of the subgraph."""
    if candidates == 0:
        return 0
    # Greedy coloring to get upper bound
    uncolored = candidates
    colors_used = 0
    while uncolored:
        # Pick a vertex, color it, remove it and neighbors
        # Use a simple greedy: pick lowest-degree vertex in remaining
        temp = uncolored
        min_deg = 999999
        pick = -1
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (conflict[v] & uncolored).bit_count()
            if deg < min_deg:
                min_deg = deg
                pick = v
            temp &= temp - 1
        if pick < 0:
            break
        colors_used += 1
        uncolored &= ~(1 << pick) & ~conflict[pick]
    return colors_used


def _mis_bnb(conflict, candidates, current_mask, current_size, best_mask, best_size, time_limit):
    """Branch and bound for maximum independent set with coloring bound."""
    if time.time() > time_limit:
        return

    if candidates == 0:
        if current_size > best_size[0]:
            best_size[0] = current_size
            best_mask[0] = current_mask
        return

    # Upper bound using coloring
    ub = _coloring_bound(conflict, candidates)
    if current_size + ub <= best_size[0]:
        return

    # Also simple bound
    if current_size + candidates.bit_count() <= best_size[0]:
        return

    # Select vertex: pick one with max degree in candidate subgraph
    temp_cand = candidates
    max_deg = -1
    best_v = -1
    while temp_cand:
        v_bit = temp_cand & -temp_cand
        v = v_bit.bit_length() - 1
        deg = (conflict[v] & candidates).bit_count()
        if deg > max_deg:
            max_deg = deg
            best_v = v
        temp_cand &= temp_cand - 1

    if best_v < 0:
        return

    v_bit = 1 << best_v

    # Branch 1: Include best_v
    new_cand_include = candidates & ~v_bit & ~conflict[best_v]
    _mis_bnb(conflict, new_cand_include, current_mask | v_bit, current_size + 1, best_mask, best_size, time_limit)

    # Branch 2: Exclude best_v
    if time.time() > time_limit:
        return
    new_cand_exclude = candidates & ~v_bit
    _mis_bnb(conflict, new_cand_exclude, current_mask, current_size, best_mask, best_size, time_limit)


def _peeling_color_bnb(conflict, m, time_limit):
    """Color by repeatedly finding Maximum Independent Sets using B&B."""
    colors = [-1] * m
    remaining_mask = (1 << m) - 1
    color_count = 0

    while remaining_mask and time.time() < time_limit:
        time_left_total = time_limit - time.time()
        if time_left_total < 0.3:
            break

        # Allocate time for this MIS
        # Use a fraction of remaining time, but ensure at least some time
        mis_time_limit = min(3.0, time_left_total * 0.6)
        if mis_time_limit < 0.1:
            break

        best_mask = [0]
        best_size = [0]

        _mis_bnb(conflict, remaining_mask, 0, 0, best_mask, best_size, time.time() + mis_time_limit)

        if best_size[0] == 0:
            # Fallback: pick one edge
            v_bit = remaining_mask & -remaining_mask
            best_mask[0] = v_bit
            best_size[0] = 1

        mask = best_mask[0]
        temp_mask = mask
        while temp_mask:
            v_bit = temp_mask & -temp_mask
            v = v_bit.bit_length() - 1
            colors[v] = color_count
            temp_mask &= temp_mask - 1

        remaining_mask &= ~mask
        color_count += 1

    # If any edges are left, color them greedily
    if remaining_mask:
        # Greedy coloring for remaining
        temp_mask = remaining_mask
        while temp_mask:
            v_bit = temp_mask & -temp_mask
            v = v_bit.bit_length() - 1
            # Find a color that doesn't conflict
            used = set()
            for nb_bit in (conflict[v] & remaining_mask):
                nb = nb_bit.bit_length() - 1
                if colors[nb] >= 0:
                    used.add(colors[nb])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c >= color_count:
                color_count = c + 1
            temp_mask &= temp_mask - 1

    return colors, color_count


def _dsatur_color(conflict, m):
    """DSATUR greedy coloring."""
    if m == 0:
        return [], 0
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
            dg = conflict[v].bit_count()
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

        for nb_bit in conflict[best_v]:
            nb = nb_bit.bit_length() - 1
            if colors[nb] == -1:
                sat[nb].add(c)

        uncolored.remove(best_v)

    return colors, color_count


def _local_search_reduce(conflict, colors, k, m, time_limit):
    """Try to reduce coloring from k to fewer colors."""
    best = list(colors)
    best_k = k

    while best_k > 1 and time.time() < time_limit:
        # Try to reduce by 1
        last_color = best_k - 1
        last_class = [v for v in range(m) if best[v] == last_color]

        if not last_class:
            best_k -= 1
            continue

        # Sort by number of available colors
        def options_count(v):
            cnt = 0
            for c in range(last_color):
                ok = True
                for nb_bit in conflict[v]:
                    nb = nb_bit.bit_length() - 1
                    if best[nb] == c:
                        ok = False
                        break
                if ok:
                    cnt += 1
            return cnt

        last_class.sort(key=options_count)
        new_colors = best[:]

        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == len(last_class):
                return True
            v = last_class[idx]
            for c in range(last_color):
                ok = True
                for nb_bit in conflict[v]:
                    nb = nb_bit.bit_length() - 1
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
            best = new_colors
            best_k -= 1
            continue

        # Try single moves: move one vertex from last class to another color
        improved = False
        for v in last_class:
            for c in range(last_color):
                ok = True
                for nb_bit in conflict[v]:
                    nb = nb_bit.bit_length() - 1
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

    # Strategy 1: DSATUR (fast initial solution)
    colors_ds, k_ds = _dsatur_color(conflict, m)
    if k_ds < best_k:
        best_colors = colors_ds
        best_k = k_ds

    # Strategy 2: B&B peeling with multiple attempts
    if _time_left() > 10.0:
        t_lim = _T0 + _DEADLINE - 8.0
        # Try B&B peeling
        colors_bnb, k_bnb = _peeling_color_bnb(conflict, m, t_lim)
        if k_bnb < best_k:
            best_colors = colors_bnb
            best_k = k_bnb

    # Local search reduction
    if _time_left() > 8.0 and best_colors is not None:
        t_lim = min(time.time() + 8.0, _T0 + _DEADLINE - 5.0)
        best_colors, best_k = _local_search_reduce(conflict, best_colors, best_k, m, t_lim)

    return best_colors


def construct():
    global _T0
    _T0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 12.0:
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
