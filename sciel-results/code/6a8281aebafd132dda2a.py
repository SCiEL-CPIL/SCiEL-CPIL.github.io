import time
import numpy as np

N = 16
NUM_EDGES = 120
NUM_K4 = 1820

# ---- Precompute edges ----
edge_to_idx = {}
idx_to_edge = []
_idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = _idx
        idx_to_edge.append((i, j))
        _idx += 1

# ---- Precompute K4s and their edge indices ----
k4_edges = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                k4_edges.append([
                    edge_to_idx[(a, b)],
                    edge_to_idx[(a, c)],
                    edge_to_idx[(a, d)],
                    edge_to_idx[(b, c)],
                    edge_to_idx[(b, d)],
                    edge_to_idx[(c, d)],
                ])

# ---- For each edge, which K4s contain it ----
edge_to_k4s = [[] for _ in range(NUM_EDGES)]
for k4_idx, edges in enumerate(k4_edges):
    for e in edges:
        edge_to_k4s[e].append(k4_idx)


def base_coloring():
    cols = np.empty(NUM_EDGES, dtype=np.int64)
    for idx, (i, j) in enumerate(idx_to_edge):
        cols[idx] = (i + j) % 17
    return cols


def compute_color_counts(colors, maxc):
    counts = np.zeros((NUM_K4, maxc + 1), dtype=np.int64)
    for k4_idx in range(NUM_K4):
        for e in k4_edges[k4_idx]:
            counts[k4_idx, colors[e]] += 1
    distinct = (counts > 0).sum(axis=1)
    bad_count = int((distinct < 5).sum())
    return counts, bad_count, distinct


def recompute_bad_for_k4s(colors, k4_idx_list, counts, distinct):
    delta_bad = 0
    for k4_idx in k4_idx_list:
        old_distinct = distinct[k4_idx]
        counts[k4_idx, :] = 0
        for e in k4_edges[k4_idx]:
            counts[k4_idx, colors[e]] += 1
        new_distinct = int((counts[k4_idx] > 0).sum())
        distinct[k4_idx] = new_distinct
        was_bad = old_distinct < 5
        is_bad = new_distinct < 5
        if was_bad and not is_bad:
            delta_bad -= 1
        elif (not was_bad) and is_bad:
            delta_bad += 1
    return delta_bad


def try_eliminate_color(colors, target_color, counts, distinct, bad_count, maxc):
    edges_of_target = np.where(colors == target_color)[0]
    if len(edges_of_target) == 0:
        return False, None, None, None, None

    used = np.unique(colors)
    candidates = [int(c) for c in used if c < target_color]
    if not candidates:
        return False, None, None, None, None

    affected_k4s = set()
    for e in edges_of_target:
        for k4 in edge_to_k4s[e]:
            affected_k4s.add(k4)
    affected_k4s = list(affected_k4s)

    def score(cand):
        s = 0
        for k4 in affected_k4s:
            if counts[k4, cand] > 0 and counts[k4, target_color] > 0:
                s += 1
        return s

    candidates.sort(key=lambda c: (-score(c), c))

    for cand in candidates:
        colors[edges_of_target] = cand
        delta = recompute_bad_for_k4s(colors, affected_k4s, counts, distinct)
        if bad_count + delta == 0:
            return True, colors, counts, distinct, 0
        else:
            colors[edges_of_target] = target_color
            recompute_bad_for_k4s(colors, affected_k4s, counts, distinct)

    return False, None, None, None, None


def local_search(colors, k, time_limit, rng):
    start = time.time()
    colors = colors.copy()
    over = np.where(colors >= k)[0]
    if len(over) > 0:
        colors[over] = rng.integers(0, k, size=len(over))

    maxc = max(k, int(colors.max()) + 1)
    counts, bad_count, distinct = compute_color_counts(colors, maxc)
    if bad_count == 0:
        return True, colors

    best_colors = colors.copy()
    best_bad = bad_count

    # Phase 1: SA
    T = 2.0
    cooling = 0.9998
    min_T = 0.05
    while time.time() - start < time_limit and T > min_T:
        e = int(rng.integers(0, NUM_EDGES))
        old = colors[e]
        new = int(rng.integers(0, k))
        if new == old:
            continue
        colors[e] = new
        affected = edge_to_k4s[e]
        delta = recompute_bad_for_k4s(colors, affected, counts, distinct)
        new_bad = bad_count + delta
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            bad_count = new_bad
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
                if bad_count == 0:
                    return True, colors
        else:
            colors[e] = old
            recompute_bad_for_k4s(colors, affected, counts, distinct)
        T *= cooling

    if best_bad < bad_count:
        colors = best_colors.copy()
        maxc = max(k, int(colors.max()) + 1)
        counts, bad_count, distinct = compute_color_counts(colors, maxc)

    if bad_count == 0:
        return True, colors

    # Phase 2: elimination
    maxc = int(colors.max()) + 1
    counts, bad_count, distinct = compute_color_counts(colors, maxc)
    for target in range(maxc - 1, 0, -1):
        if time.time() - start > time_limit:
            break
        if (colors == target).sum() == 0:
            continue
        ok, colors2, counts2, distinct2, bad2 = try_eliminate_color(
            colors, target, counts, distinct, bad_count, maxc)
        if ok:
            colors, counts, distinct, bad_count = colors2, counts2, distinct2, bad2

    if bad_count == 0:
        return True, colors

    # Phase 3: SA polish
    maxc = int(colors.max()) + 1
    k2 = maxc
    T = 1.0
    cooling = 0.9998
    min_T = 0.03
    while time.time() - start < time_limit and T > min_T:
        e = int(rng.integers(0, NUM_EDGES))
        old = colors[e]
        new = int(rng.integers(0, k2))
        if new == old:
            continue
        colors[e] = new
        affected = edge_to_k4s[e]
        delta = recompute_bad_for_k4s(colors, affected, counts, distinct)
        new_bad = bad_count + delta
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            bad_count = new_bad
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
                if bad_count == 0:
                    return True, colors
        else:
            colors[e] = old
            recompute_bad_for_k4s(colors, affected, counts, distinct)
        T *= cooling

    if best_bad == 0:
        return True, best_colors
    return False, None


def construct():
    total_start = time.time()
    deadline = 140.0

    rng = np.random.default_rng(12345)
    base = base_coloring()

    best_valid = None
    for k in [13, 14, 15]:
        if time.time() - total_start > deadline:
            break
        remaining = deadline - (time.time() - total_start)
        if remaining < 8:
            break
        attempts = 0
        found = False
        while time.time() - total_start < deadline - 5 and not found and attempts < 4:
            attempts += 1
            t_left = deadline - 5 - (time.time() - total_start)
            t_left = max(3.0, t_left / max(1, 4 - attempts))
            ok, colors = local_search(base, k, t_left, rng)
            if ok:
                best_valid = colors
                found = True
                break
            if not found:
                base = base.copy()
                pert = rng.integers(0, 17, size=NUM_EDGES)
                base = np.where(rng.random(size=NUM_EDGES) < 0.1, pert, base)

        if best_valid is not None:
            break

    if best_valid is not None:
        cols = best_valid
        uniq = np.unique(cols)
        remap = {int(u): i for i, u in enumerate(uniq)}
        out = np.array([remap[int(c)] for c in cols], dtype=int)
        for k4_idx in range(NUM_K4):
            if len(np.unique(out[k4_edges[k4_idx]])) < 5:
                raise ValueError("invalid")
        return list(out)

    out = base
    for k4_idx in range(NUM_K4):
        if len(np.unique(out[k4_edges[k4_idx]])) < 5:
            out = np.array([(i + j) % 17 for i, j in idx_to_edge], dtype=int)
            break
    return list(out)
