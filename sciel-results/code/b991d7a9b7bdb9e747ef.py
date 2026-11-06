"""Strong edge colouring: multi-strategy DSATUR + induced matching peeling."""
import time
from random import Random

_T0 = time.time()
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_bitset_conflict(edges, n):
    """Build conflict graph as list of integers (bitsets)."""
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


def _mis_bnb(conflict_bitsets, deadline):
    """
    Branch-and-bound Maximum Independent Set on conflict graph.
    Returns a list of edge indices forming the largest induced matching.
    Uses bitset representation for fast neighbor checks.
    """
    m = len(conflict_bitsets)
    if m == 0:
        return []
    
    # Convert to list of sets for easier manipulation during search
    # But use bitsets for fast intersection
    # We'll work with a "remaining" set as a bitset
    
    best_size = 0
    best_set = set()
    
    # Upper bound: greedy coloring of the remaining graph gives upper bound on MIS
    # We'll use a simple greedy coloring bound
    
    def greedy_color_bound(subset_bits):
        """Color the subgraph induced by subset_bits, return number of colors used."""
        remaining = subset_bits
        num_colors = 0
        while remaining:
            num_colors += 1
            # Greedy: pick a vertex, remove it and its neighbors
            # Pick vertex with max degree in remaining for better bound
            best_v = -1
            best_deg = -1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (conflict_bitsets[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v >= 0:
                remaining &= ~(conflict_bitsets[best_v] | (1 << best_v))
        return num_colors
    
    def bnb(remaining_bits, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if remaining_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + greedy_color_bound(remaining_bits)
        ub = current_size + greedy_color_bound(remaining_bits)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with max degree in remaining for better pruning
        best_v = -1
        best_deg = -1
        temp = remaining_bits
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = (conflict_bitsets[v] & remaining_bits).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: include v
        new_remaining = remaining_bits & ~(conflict_bitsets[v] | v_bit)
        bnb(new_remaining, current_set + [v], current_size + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: exclude v
        bnb(remaining_bits & ~v_bit, current_set, current_size)
    
    bnb((1 << m) - 1, [], 0)
    return list(best_set)


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
    """Find a large induced matching among remaining edges using BnB."""
    m = len(edges)
    if not remaining:
        return []
    
    rem_set = set(remaining)
    
    # Build bitset conflict restricted to remaining edges
    # Map remaining indices to compact indices
    rem_list = sorted(remaining)
    rem_to_compact = {v: i for i, v in enumerate(rem_list)}
    k = len(rem_list)
    
    if k == 0:
        return []
    
    # Build compact conflict bitsets
    compact_conflict = [0] * k
    for i, orig_i in enumerate(rem_list):
        for j, orig_j in enumerate(rem_list):
            if i != j and orig_j in conflict[orig_i]:
                compact_conflict[i] |= (1 << j)
    
    # Use BnB with time limit
    deadline = min(time.time() + 2.0, _T0 + _DEADLINE - 5.0)
    compact_mis = _mis_bnb(compact_conflict, deadline)
    
    # Map back to original indices
    return [rem_list[i] for i in compact_mis]


def _peeling_color(edges, n, conflict, rng, time_limit):
    """Color by repeatedly finding large induced matchings."""
    m = len(edges)
    colors = [0] * m
    remaining = set(range(m))
    color_count = 0

    while remaining and time.time() < time_limit:
        match = _find_large_induced_matching(edges, n, conflict, list(remaining), rng)
        if not match:
            # Fallback: single edge
            match = [next(iter(remaining))]
        for i in match:
            colors[i] = color_count
            remaining.discard(i)
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
    conflict_bits = _build_bitset_conflict(edges, n)

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

    # Strategy 3: Peeling with BnB MIS
    if _time_left() > 25.0:
        t_lim = min(time.time() + 15.0, _T0 + _DEADLINE - 12.0)
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
