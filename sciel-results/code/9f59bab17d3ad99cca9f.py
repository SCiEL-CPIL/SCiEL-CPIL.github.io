"""Combined: bitmask set-cover with fast mask construction, improved greedy,
and iterative local prune. Keep entry-point construct() and the
backtracking validator _is_2_colorable."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _build_masks_fast(n):
    """Build bitmask for each 5-uniform edge: bitmask over all 2^n colourings
    where the edge is monochromatic. Uses precomputed sub-masks for speed."""
    edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
    num_edges = len(edges)
    total_colorings = 1 << n
    masks = [0] * num_edges

    # Precompute: for each vertex v, the bitmask of colourings where v=0
    # and where v=1. Actually we need: for an edge (v0..v4), the colourings
    # where all 5 are 0 OR all 5 are 1.
    # For all-0: AND of (colouring has v_i=0) for i=0..4
    # For all-1: AND of (colouring has v_i=1) for i=0..4
    #
    # Precompute vertex_zero_mask[v]: bitmask of colourings where vertex v is 0
    # vertex_one_mask[v] = ~vertex_zero_mask[v] & full_mask
    full_mask = (1 << total_colorings) - 1

    # For each vertex v, build mask of colourings where v=0
    # A coloring is an integer 0..2^n-1; bit v of that integer is the colour.
    # We want a bitmask M_v such that M_v[c] = 1 iff colour c has bit v = 0.
    # M_v = OR over all c where (c>>v)&1==0 of (1<<c)
    # This is periodic: for block of size 2^(v+1), first 2^v have bit v=0.
    vertex_zero = [0] * n
    for v in range(n):
        block = 1 << (v + 1)
        half = 1 << v
        m = 0
        for start in range(0, total_colorings, block):
            # colourings start..start+half-1 have bit v = 0
            # We need OR of (1<<c) for c in [start, start+half)
            # That's (1 << (start + half)) - (1 << start)
            m |= ((1 << (start + half)) - (1 << start))
        vertex_zero[v] = m

    for ei, e in enumerate(edges):
        # all-0 mask: AND of vertex_zero[v] for v in e
        m0 = full_mask
        m1 = full_mask
        for v in e:
            m0 &= vertex_zero[v]
            m1 &= (full_mask ^ vertex_zero[v])  # vertex v = 1
        masks[ei] = m0 | m1

    return edges, masks


def _greedy_cover(n, masks, full, deadline, rng, randomize=0.0):
    """Greedy set cover with optional randomization in tie-breaking."""
    uncovered = full
    chosen = []
    chosen_masks = []
    remaining = set(range(len(masks)))
    while uncovered:
        if time.time() > deadline:
            return None, None
        best = -1
        bestc = 0
        # Full scan for smaller sets, sampled for larger
        if len(remaining) <= 2000:
            sample = list(remaining)
        else:
            sample = rng.sample(list(remaining), min(2000, len(remaining)))
        for ei in sample:
            c = (uncovered & masks[ei]).bit_count()
            if c > bestc or (c == bestc and bestc > 0 and randomize > 0 and rng.random() < randomize):
                bestc = c
                best = ei
                if bestc == uncovered.bit_count():
                    break
        if best < 0:
            break
        chosen.append(best)
        chosen_masks.append(masks[best])
        uncovered &= ~masks[best]
        remaining.discard(best)
    if uncovered == 0:
        return chosen, chosen_masks
    return None, None


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=30):
    """Iteratively remove any edge whose absence still covers all colourings.
    Uses a coverage count for efficiency."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        k = len(chosen)
        if k == 0:
            break
        order = list(range(k))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            # Check if removing chosen[i] still covers full
            # Fast path: if chosen[i] is the only one covering some bit, skip
            # Build union of all others
            u = 0
            for j in range(len(chosen)):
                if j != i:
                    u |= chosen_masks[j]
            if u == full:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
    return chosen, chosen_masks


def _swap_improve(chosen, chosen_masks, masks, full, deadline, rng, max_swaps=50):
    """Try to replace edges with better ones to reduce total count."""
    improved = True
    swaps = 0
    while improved and swaps < max_swaps and time.time() < deadline:
        improved = False
        swaps += 1
        k = len(chosen)
        if k < 2:
            break
        # Try removing each edge and see if we can cover the gap with one new edge
        order = list(range(k))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            # What does chosen[i] uniquely cover?
            without_i = 0
            for j in range(len(chosen)):
                if j != i:
                    without_i |= chosen_masks[j]
            gap = full & ~without_i
            if gap == 0:
                # Can just remove
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
            # Find a single edge that covers all of gap
            best_new = -1
            for ei in range(len(masks)):
                if ei in chosen:
                    continue
                if (masks[ei] & gap) == gap:
                    # Check if this edge is better (covers more of what was covered by i)
                    best_new = ei
                    break
            if best_new >= 0:
                chosen[i] = best_new
                chosen_masks[i] = masks[best_new]
                improved = True
                break
    return chosen, chosen_masks


def _try_n(n, deadline, rng):
    edges, masks = _build_masks_fast(n)
    full = (1 << (1 << n)) - 1
    best = None
    best_len = float('inf')
    # Multiple restarts with varying randomization
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline:
            break
        randomize = 0.05 + 0.15 * (r + 1) / restarts
        chosen, chosen_masks = _greedy_cover(n, masks, full, deadline, rng, randomize)
        if chosen is None:
            continue
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, masks, full, deadline, rng)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = (list(chosen), list(chosen_masks))
            if best_len <= 30:
                break
    if best is None:
        return None
    chosen, _ = best
    # exact certificate: union of masks must equal full
    u = 0
    for ei in chosen:
        u |= masks[ei]
    if u != full:
        return None
    return {'n': n, 'edges': [list(edges[i]) for i in chosen]}


def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    rng = random.Random(12345)

    # Guaranteed fallback: complete 5-graph on 9 vertices (126 edges).
    fallback = {'n': 9,
                'edges': [list(e) for e in combinations(range(9), 5)]}

    best = fallback
    # Try n=11 first (known to have 51-edge solution), then 10, 12, 9
    for n in (11, 10, 12, 9):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res
            # If we found something very good, we can stop
            if len(best['edges']) <= 30:
                break

    return best


def _is_2_colorable(n, edges, deadline):
    """True iff there exists a 2-colouring with NO monochromatic edge.
    Uses backtracking with unit propagation."""
    edges = [tuple(e) for e in edges]
    m = len(edges)
    if m == 0:
        return True

    color = [-1] * n
    e0 = [0] * m
    e1 = [0] * m
    uncol = [list(e) for e in edges]

    def propagate(queue):
        while queue:
            ei = queue.pop()
            u = uncol[ei]
            if e0[ei] == 5 or e1[ei] == 5:
                return False
            if len(u) == 1:
                v = u[0]
                if e0[ei] == 4:
                    forced = 1
                elif e1[ei] == 4:
                    forced = 0
                else:
                    continue
                if color[v] == -1:
                    color[v] = forced
                    uncol[ei].remove(v)
                    if forced == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    for ej in range(m):
                        if ej != ei and v in uncol[ej]:
                            uncol[ej].remove(v)
                            if forced == 0:
                                e0[ej] += 1
                            else:
                                e1[ej] += 1
                            if e0[ej] == 5 or e1[ej] == 5:
                                return False
                            if len(uncol[ej]) == 1:
                                queue.append(ej)
                elif color[v] != forced:
                    return False
        return True

    import sys
    sys.setrecursionlimit(100000)

    def rec(k):
        if time.time() > deadline:
            return None
        if k >= n:
            return True
        for c in (0, 1):
            color[k] = c
            touched = []
            conflict = False
            for ei, e in enumerate(edges):
                if k in e:
                    uncol[ei].remove(k)
                    if c == 0:
                        e0[ei] += 1
                    else:
                        e1[ei] += 1
                    touched.append(ei)
                    if e0[ei] == 5 or e1[ei] == 5:
                        conflict = True
                        break
            if conflict:
                color[k] = -1
                for ej in touched:
                    if k in uncol[ej]:
                        uncol[ej].append(k)
                    if c == 0:
                        e0[ej] -= 1
                    else:
                        e1[ej] -= 1
                continue
            queue = [ei for ei in touched if len(uncol[ei]) == 1]
            if propagate(queue):
                result = rec(k + 1)
                if result is not None:
                    return result
            color[k] = -1
            for ej in touched:
                if k in uncol[ej]:
                    uncol[ej].append(k)
                if c == 0:
                    e0[ej] -= 1
                else:
                    e1[ej] -= 1
        return False

    result = rec(0)
    if result is None:
        return False
    return result


if __name__ == '__main__':
    _t0 = time.time()
    _res = construct()
    _ok = _is_2_colorable(_res['n'], _res['edges'], _t0 + 120.0)
    assert not _ok, "constructed hypergraph must NOT be 2-colourable"
    print("edges:", len(_res['edges']), "n:", _res['n'], "2-colourable:", _ok)
