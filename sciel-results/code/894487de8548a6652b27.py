"""Edge-colourings of K16 in which every K4 sees five colours."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 145.0  # 150s wall-clock limit, 5s safety margin
    E = n * (n - 1) // 2  # 120 edges

    # Map (i,j) i<j to a flat edge index 0..119
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)

    # Build list of edges (i,j) in lexicographic order
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    # edge_list[e] should match edge_idx(e)
    assert len(edge_list) == E

    # Precompute all K4s as tuples of 6 edge indices
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = (
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d),
                    )
                    k4_edges.append(edges)
    num_k4s = len(k4_edges)  # 1820

    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4i, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4i)

    # ---------- fast per-K4 distinct count using bitmasks (colours < 64) ----------
    def distinct_count(edges, colors):
        # returns number of distinct colours among the 6 edges
        seen = 0
        d = 0
        for e in edges:
            m = 1 << colors[e]
            if not (seen & m):
                seen |= m
                d += 1
        return d

    def is_valid(colors):
        for edges in k4_edges:
            if distinct_count(edges, colors) < 5:
                return False
        return True

    def penalty(colors):
        # sum of (5 - distinct)^2 over violating K4s
        total = 0
        for edges in k4_edges:
            d = distinct_count(edges, colors)
            if d < 5:
                total += (5 - d) * (5 - d)
        return total

    def num_colours(colors):
        seen = 0
        for c in colors:
            seen |= (1 << c)
        return bin(seen).count('1')

    # ---------- fallback: 17-colour cyclic construction ----------
    def make_fallback():
        colors = [0] * E
        for e, (i, j) in enumerate(edge_list):
            colors[e] = (i + j) % 17
        return colors

    best_colors = None
    best_score = 10**9

    def consider(colors):
        nonlocal best_colors, best_score
        if not is_valid(colors):
            return
        # score = number of colours used (minimize)
        sc = num_colours(colors)
        if sc < best_score:
            best_score = sc
            best_colors = list(colors)

    # Seed with the known valid 17-colour construction
    fb = make_fallback()
    consider(fb)

    # ============================================================
    # Phase 1: Direct edge-level local search with delta cost
    # ============================================================
    def edge_local_search(k, time_limit):
        """Random-restart hill-climbing on edge colours 0..k-1.
        Objective: minimise penalty (0 = valid), then minimise colour count."""
        local_start = time.time()
        best_local = None
        best_local_score = 10**9

        for restart in range(200):
            if time.time() > time_limit:
                break
            colors = np.random.randint(0, k, size=E, dtype=np.int64)
            # per-K4 penalty cache
            kpen = np.zeros(num_k4s, dtype=np.int64)
            cur_pen = 0
            for k4i, edges in enumerate(k4_edges):
                d = distinct_count(edges, colors)
                if d < 5:
                    p = (5 - d) * (5 - d)
                    kpen[k4i] = p
                    cur_pen += p

            if cur_pen == 0:
                sc = num_colours(colors)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = colors.copy()

            # hill climb: reduce penalty
            for it in range(4000):
                if time.time() > time_limit:
                    break
                if cur_pen == 0:
                    break
                # pick a violating K4 (random among violators)
                viol = np.nonzero(kpen > 0)[0]
                if len(viol) == 0:
                    break
                k4i = int(viol[random.randrange(len(viol))])
                edges = k4_edges[k4i]
                e = edges[random.randrange(6)]
                old = int(colors[e])
                # try a new colour
                new = random.randrange(k)
                if new == old:
                    continue
                colors[e] = new
                # recompute penalty only for K4s containing e
                delta = 0
                for k4j in edge_to_k4s[e]:
                    newp = kpen[k4j]
                    d = distinct_count(k4_edges[k4j], colors)
                    if d < 5:
                        newp = (5 - d) * (5 - d)
                    delta += newp - kpen[k4j]
                    kpen[k4j] = newp
                cur_pen += delta
                if delta > 0:
                    # revert
                    colors[e] = old
                    for k4j in edge_to_k4s[e]:
                        d = distinct_count(k4_edges[k4j], colors)
                        kpen[k4j] = (5 - d) * (5 - d) if d < 5 else 0
                    cur_pen -= delta
                # track best valid
                if cur_pen == 0:
                    sc = num_colours(colors)
                    if sc < best_local_score:
                        best_local_score = sc
                        best_local = colors.copy()

            if time.time() > time_limit:
                break

        return best_local, best_local_score

    # Try a few small k values for direct search
    for k in (13, 14, 15, 16):
        if time.time() > deadline - 15:
            break
        tl = min(deadline - 15, time.time() + 25)
        res, sc = edge_local_search(k, tl)
        if res is not None:
            consider(res)

    # ============================================================
    # Phase 2: Vertex-labeling search (structural family)
    # Edge colour = f(label[i], label[j]) with small label set.
    # This drastically reduces the search space (16 labels vs 120 edges).
    # ============================================================
    def edge_colors_from_labels(labels, func, param):
        colors = [0] * E
        for e, (i, j) in enumerate(edge_list):
            li, lj = labels[i], labels[j]
            if func == 'sum':
                c = (li + lj) % param
            elif func == 'prod':
                c = (li * lj) % param
            elif func == 'xor':
                c = li ^ lj
            elif func == 'sumprod':
                c = (li * lj + li + lj) % param
            else:
                c = (li + lj) % param
            colors[e] = c
        return colors

    def label_local_search(func, param, label_range, time_limit):
        """Hill-climb on vertex labels to minimise penalty, then colour count."""
        local_start = time.time()
        best_local = None
        best_local_score = 10**9

        for restart in range(300):
            if time.time() > time_limit:
                break
            labels = [random.randrange(label_range) for _ in range(n)]
            colors = edge_colors_from_labels(labels, func, param)
            cur_pen = penalty(colors)

            if cur_pen == 0:
                sc = num_colours(colors)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = colors.copy()

            for it in range(3000):
                if time.time() > time_limit:
                    break
                if cur_pen == 0:
                    break
                v = random.randrange(n)
                old = labels[v]
                new = random.randrange(label_range)
                if new == old:
                    continue
                labels[v] = new
                colors = edge_colors_from_labels(labels, func, param)
                new_pen = penalty(colors)
                if new_pen > cur_pen:
                    labels[v] = old
                    colors = edge_colors_from_labels(labels, func, param)
                else:
                    cur_pen = new_pen
                    if cur_pen == 0:
                        sc = num_colours(colors)
                        if sc < best_local_score:
                            best_local_score = sc
                            best_local = colors.copy()

            if time.time() > time_limit:
                break

        return best_local, best_local_score

    configs = [
        ('sum', 13, 5),
        ('sum', 13, 6),
        ('sum', 14, 5),
        ('sum', 15, 5),
        ('sum', 16, 6),
        ('prod', 13, 5),
        ('prod', 17, 6),
        ('xor', 16, 4),
        ('xor', 32, 5),
        ('sumprod', 13, 5),
        ('sumprod', 15, 5),
    ]
    for func, param, lr in configs:
        if time.time() > deadline - 10:
            break
        tl = min(deadline - 10, time.time() + 12)
        res, sc = label_local_search(func, param, lr, tl)
        if res is not None:
            consider(res)

    # ============================================================
    # Phase 3: Final polish pass - tighten the best valid solution
    # Try to reduce colour count further by remapping/re-colouring.
    # ============================================================
    if best_colors is not None:
        base = list(best_colors)
        cur_sc = best_score
        # attempt to eliminate unused colours by merging, or recolor edges
        # that are 'redundant' (their colour appears only once and K4s still valid)
        # Simple: local search minimising colour count while keeping valid.
        polish_start = time.time()
        polish_limit = min(deadline - 2, polish_start + 20)
        colors = base[:]
        # compute per-colour usage
        while time.time() < polish_limit:
            # count colours
            counts = {}
            for c in colors:
                counts[c] = counts.get(c, 0) + 1
            # find a colour with minimal usage (candidate to eliminate)
            if not counts:
                break
            # pick a colour to try to eliminate (lowest count, not the only one)
            candidates = [c for c, ct in counts.items() if ct > 0]
            if len(candidates) <= 1:
                break
            # try to recolor edges of a chosen colour to other colours
            target = min(candidates, key=lambda c: counts[c])
            improved = False
            for e in range(E):
                if colors[e] != target:
                    continue
                old = colors[e]
                # try reassigning to another existing colour
                for new in candidates:
                    if new == target:
                        continue
                    colors[e] = new
                    if is_valid(colors):
                        sc = num_colours(colors)
                        if sc < cur_sc:
                            cur_sc = sc
                            best_score = sc
                            best_colors = list(colors)
                            improved = True
                        break
                if improved:
                    break
            if not improved:
                # no single-edge improvement; break to avoid loops
                break

    # Final safety: ensure we return a valid coloring
    if best_colors is None:
        best_colors = make_fallback()
    else:
        if not is_valid(best_colors):
            best_colors = make_fallback()

    # All colours must be in 0..119
    result = []
    for c in best_colors:
        c = int(c) % 120
        result.append(c)
    return result

# EVOLVE-BLOCK-END
