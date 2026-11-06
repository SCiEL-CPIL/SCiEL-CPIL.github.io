"""Combined: precomputed-AP simulated annealing (inspiration 1) + color-reduction
local search / try-remove (current program). Entry point: construct()."""
import time
import random
import math

N = 128


def _build_aps():
    aps = []
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 < N:
                aps.append((a, a + d, a + 2 * d, i3))
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    return aps, pos_to_aps


def _ap_violation(ap, colors):
    return len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3


def _count_violations(colors, aps):
    return sum(1 for ap in aps if _ap_violation(ap, colors))


def _check_valid(colors, aps):
    return _count_violations(colors, aps) == 0


def _count_colors(colors):
    return len(set(colors))


def _greedy_valid(colors_init, deadline, aps, pos_to_aps, max_colors=128):
    """Fill/fix a partial coloring greedily, small colors first. Returns None if stuck."""
    colors = list(colors_init)
    for x in range(N):
        if time.time() > deadline:
            return None
        if colors[x] is not None:
            continue
        found = False
        for c in range(max_colors):
            colors[x] = c
            valid = True
            for ap_idx in pos_to_aps[x]:
                ap = aps[ap_idx]
                if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return colors


def _sim_anneal(k, deadline, aps, pos_to_aps, seed, max_steps=200000):
    """Simulated annealing over palette [0,k). Returns (best_colors, best_violations)."""
    rng = random.Random(seed)
    colors = [rng.randint(0, k - 1) for _ in range(N)]
    cur_v = _count_violations(colors, aps)
    best_colors = colors[:]
    best_v = cur_v
    if cur_v == 0:
        return best_colors, 0
    temp = 1.0
    cooling = 0.9999
    steps = 0
    while steps < max_steps and time.time() < deadline:
        steps += 1
        i = rng.randrange(N)
        old = colors[i]
        new = rng.randint(0, k - 1)
        if new == old:
            continue
        delta = 0
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            was = _ap_violation(ap, colors)
            colors[i] = new
            now = _ap_violation(ap, colors)
            colors[i] = old
            if was and not now:
                delta -= 1
            elif not was and now:
                delta += 1
        nv = cur_v + delta
        if nv <= cur_v or rng.random() < math.exp(-(nv - cur_v) / temp):
            colors[i] = new
            cur_v = nv
            if cur_v < best_v:
                best_v = cur_v
                best_colors = colors[:]
                if cur_v == 0:
                    break
        temp *= cooling
    return best_colors, best_v


def _try_remove_color(colors, target_color, aps, pos_to_aps):
    """Eliminate target_color by recoloring its positions to other colors."""
    other_colors = sorted(set(colors) - {target_color})
    if not other_colors:
        return None
    positions = [i for i in range(N) if colors[i] == target_color]
    if not positions:
        return list(colors)
    new_colors = list(colors)
    # Most constrained positions first
    pos_constraint = []
    for pos in positions:
        pos_constraint.append((len(pos_to_aps[pos]), pos))
    pos_constraint.sort(reverse=True)
    for _, pos in pos_constraint:
        found = False
        for new_color in other_colors:
            new_colors[pos] = new_color
            valid = True
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                if len({new_colors[ap[0]], new_colors[ap[1]], new_colors[ap[2]], new_colors[ap[3]]}) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
        if not found:
            return None
    return new_colors


def _local_search_reduce_colors(colors, deadline, aps, pos_to_aps, max_attempts=200):
    """Reduce distinct colors by repeatedly removing a color and, if stuck, local moves."""
    colors = list(colors)
    best = _count_colors(colors)
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        unique = sorted(set(colors))
        if len(unique) <= 2:
            break
        improved = False
        rng = random.Random(attempt * 7919 + 13)
        color_order = list(unique)
        rng.shuffle(color_order)
        for target in color_order:
            if time.time() > deadline:
                break
            new_colors = _try_remove_color(colors, target, aps, pos_to_aps)
            if new_colors is not None and _check_valid(new_colors, aps):
                cnt = _count_colors(new_colors)
                if cnt < best:
                    colors = new_colors
                    best = cnt
                    improved = True
                    break
        if not improved:
            # Try a few random single-position recolors to enable further removals
            for _ in range(50):
                if time.time() > deadline:
                    break
                pos = rng.randrange(N)
                old = colors[pos]
                for nc in set(colors) - {old}:
                    colors[pos] = nc
                    if _check_valid(colors, aps):
                        break
                    colors[pos] = old
    return colors


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin under 90s
    aps, pos_to_aps = _build_aps()

    best_solution = None
    best_count = N

    # Phase 1: simulated annealing over increasing palette sizes,
    # then reduce colors. This is the primary search.
    for k in range(3, 25):
        if time.time() > deadline - 12:
            break
        restarts = 4 if k >= 10 else 2
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            cols, v = _sim_anneal(k, deadline - 10, aps, pos_to_aps, seed=r * 100000 + k)
            if v == 0:
                cnt = _count_colors(cols)
                if cnt < best_count:
                    best_count = cnt
                    best_solution = list(cols)
                if best_count == 3:
                    break
        if best_count == 3:
            break

    # Phase 2: if nothing valid, greedy fallback
    if best_solution is None:
        for k in range(3, 20):
            if time.time() > deadline - 15:
                break
            cols = _greedy_valid([None] * N, deadline - 15, aps, pos_to_aps, max_colors=k)
            if cols is not None and _check_valid(cols, aps):
                best_solution = cols
                best_count = _count_colors(cols)
                break

    # Phase 3: reduce colors further with local search
    if best_solution is not None and time.time() < deadline - 6:
        improved = _local_search_reduce_colors(best_solution, deadline - 6, aps, pos_to_aps)
        if improved is not None and _check_valid(improved, aps):
            if _count_colors(improved) <= best_count:
                best_solution = improved
                best_count = _count_colors(best_solution)

    # Emergency fallback
    if best_solution is None:
        return list(range(N))

    return best_solution
