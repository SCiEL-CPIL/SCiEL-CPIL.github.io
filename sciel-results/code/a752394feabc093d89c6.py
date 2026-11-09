"""Stronger design: conflict graph + Tomita B&B (color bound) + local improvement, time-guarded."""
import time

_N = 256
_FULL = (1 << _N) - 1

def _build_graph():
    adj = [0] * _N
    for i in range(_N):
        a = i + 1
        for j in range(i + 1, _N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _greedy_iset(adj, remaining):
    """Greedy maximal independent set from 'remaining' mask. Returns (mask, count)."""
    chosen = 0
    rem = remaining
    while rem:
        # pick vertex with minimum degree in rem
        best_v = -1
        best_deg = _N + 1
        m = rem
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            deg = (rem & adj[idx]).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = idx
            m &= ~lsb
        chosen |= (1 << best_v)
        rem &= ~(1 << best_v)
        rem &= ~adj[best_v]
    return chosen, chosen.bit_count()

def _color_bound(cand, adj):
    """Greedy coloring of subgraph induced by cand; return (colors_list, ub)."""
    rem = cand
    colors = []
    while rem:
        cls = 0
        m = rem
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            cls |= lsb
            m &= ~lsb
            m &= ~adj[idx]
        colors.append(cls.bit_count())
        rem &= ~cls
    return colors, colors[-1]

def _tomita(cand, adj, depth, best_ref, deadline):
    if cand == 0:
        return
    if time.time() > deadline:
        return
    colors, ub = _color_bound(cand, adj)
    if depth + ub <= best_ref[0]:
        return
    # order vertices by descending degree in cand
    verts = []
    m = cand
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        verts.append(( (cand & adj[idx]).bit_count(), idx ))
        m &= ~lsb
    verts.sort(reverse=True)
    for deg, v in verts:
        if depth + 1 + ub <= best_ref[0]:
            break
        # include v
        if depth + 1 > best_ref[0]:
            best_ref[0] = depth + 1
        _tomita(cand & ~adj[v] & ~(1 << v), adj, depth + 1, best_ref, deadline)
        cand &= ~(1 << v)
        ub -= 1
        if depth + ub <= best_ref[0]:
            break

def _local_improve(chosen_mask, adj, deadline):
    """Add any non-conflicting vertex; then 1-for-2 swaps. Returns improved mask."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # add
        m = _FULL & ~chosen_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            if (chosen_mask & adj[idx]) == 0:
                chosen_mask |= lsb
                improved = True
                break
            m &= ~lsb
        if improved:
            continue
        # swap: remove v, add two
        m = chosen_mask
        while m and time.time() < deadline:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            rest = chosen_mask & ~lsb
            # find two addable non-conflicting-with-rest vertices
            a1 = -1
            a2 = -1
            m2 = _FULL & ~lsb
            while m2:
                l2 = m2 & (-m2)
                i2 = l2.bit_length() - 1
                if (rest & adj[i2]) == 0:
                    if a1 == -1:
                        a1 = i2
                    elif (adj[a1] & l2) == 0:
                        a2 = i2
                        break
                m2 &= ~l2
            if a1 != -1 and a2 != -1:
                chosen_mask = rest | (1 << a1) | (1 << a2)
                improved = True
                break
            m &= ~lsb
    return chosen_mask

def construct():
    start = time.time()
    deadline = start + 80.0
    adj = _build_graph()

    # greedy seed
    seed_mask, seed_size = _greedy_iset(adj, _FULL)
    best_ref = [seed_size]

    # Tomita
    _tomita(_FULL, adj, 0, best_ref, deadline)

    # local improve on seed (and on best if we tracked it; seed is our concrete set)
    improved_mask = _local_improve(seed_mask, adj, deadline)
    if improved_mask.bit_count() > best_ref[0]:
        best_ref[0] = improved_mask.bit_count()

    # produce concrete set from improved_mask if it's the best, else from seed
    final_mask = improved_mask if improved_mask.bit_count() >= seed_size else seed_mask
    out = []
    m = final_mask
    while m:
        lsb = m & (-m)
        out.append(lsb.bit_length())  # 1-indexed value
        m &= ~lsb
    return out
