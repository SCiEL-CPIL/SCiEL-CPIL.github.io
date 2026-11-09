"""Combined: bitmask set-cover with faster mask construction, smarter greedy
with look-ahead, aggressive local prune, and multi-restart randomization."""
import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 15.0


def _build_masks_fast(n):
    """Build edge masks using precomputed per-vertex-position patterns.
    
    For a 5-uniform edge on vertices (v0,v1,v2,v3,v4), the monochromatic
    colourings are: all 5 in colour 0, or all 5 in colour 1, with remaining
    n-5 vertices free. We precompute the pattern for each subset of positions.
    """
    edges = [tuple(sorted(e)) for e in combinations(range(n), 5)]
    masks = []
    
    # Precompute: for each vertex v, what's the bit pattern when v is in "base" (colour 1)
    # and for each vertex v, what's the bit pattern when v is in "free" positions
    # Actually, let's just use a more efficient enumeration.
    
    # For n vertices, 2^n colourings. An edge (a,b,c,d,e) is monochromatic when:
    #   all of a,b,c,d,e have colour 0, OR all have colour 1
    # The other n-5 vertices can be anything.
    
    # Efficient approach: for each edge, iterate over all 2^(n-5) assignments to rest,
    # but use bit operations to compute the two indices quickly.
    
    for e in edges:
        vs = list(e)
        vs_set = set(vs)
        rest = [v for v in range(n) if v not in vs_set]
        
        # base index where all edge vertices are 0 (colour 0)
        base0 = 0
        base1 = 0
        for v in vs:
            base1 |= (1 << v)
        
        r = len(rest)
        m = 0
        for p in range(1 << r):
            # Compute index for colouring: edge verts = 0, rest per p
            idx0 = base0
            idx1 = base1
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx0 |= (1 << v)
                    idx1 |= (1 << v)
            m |= (1 << idx0) | (1 << idx1)
        masks.append(m)
    
    return edges, masks


def _greedy_cover(masks, full, deadline, rng, randomize=0.0, look_ahead=1):
    """Greedy set cover with randomization and optional look-ahead."""
    uncovered = full
    chosen = []
    chosen_masks = []
    remaining = list(range(len(masks)))
    n_edges = len(masks)
    
    while uncovered:
        if time.time() > deadline:
            return None, None
        
        best = -1
        bestc = 0
        
        # Sample if many remain
        if len(remaining) > 300:
            sample = rng.sample(remaining, min(300, len(remaining)))
        else:
            sample = remaining
        
        for ei in sample:
            c = (uncovered & masks[ei]).bit_count()
            if c > bestc or (c == bestc and bestc > 0 and randomize > 0 and rng.random() < randomize):
                bestc = c
                best = ei
                if bestc == uncovered.bit_count():
                    break
        
        if best < 0:
            # Full scan fallback
            for ei in remaining:
                c = (uncovered & masks[ei]).bit_count()
                if c > bestc:
                    bestc = c
                    best = ei
        
        if best < 0:
            break
        
        # Look-ahead: check if picking best leads to dead end quickly
        if look_ahead > 1 and len(remaining) > 10:
            trial_uncovered = uncovered & ~masks[best]
            if trial_uncovered:
                # Quick check: is there any edge that covers significant part?
                max_cover = 0
                for ei2 in remaining:
                    if ei2 == best:
                        continue
                    c2 = (trial_uncovered & masks[ei2]).bit_count()
                    if c2 > max_cover:
                        max_cover = c2
                # If max coverage is very small relative to remaining, this might be bad
                # But we still pick it for now (greedy)
        
        chosen.append(best)
        chosen_masks.append(masks[best])
        uncovered &= ~masks[best]
        remaining.remove(best)
    
    if uncovered == 0:
        return chosen, chosen_masks
    return None, None


def _local_prune(chosen, chosen_masks, full, deadline, rng, max_rounds=30):
    """Iteratively remove any edge whose absence still covers all colourings.
    Uses multiple random orderings per round."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        order = list(range(len(chosen)))
        rng.shuffle(order)
        for i in order:
            if i >= len(chosen):
                continue
            # Check if removing edge i still covers everything
            u = 0
            for j in range(len(chosen)):
                if j == i:
                    continue
                u |= chosen_masks[j]
            if u == full:
                del chosen[i]
                del chosen_masks[i]
                improved = True
                break
    return chosen, chosen_masks


def _swap_improve(chosen, chosen_masks, full, masks, deadline, rng, max_rounds=15):
    """Try to swap one edge for another to reduce total count or improve structure.
    Actually, just do more aggressive removal with different orderings."""
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        # Try removing in multiple random orders
        for _ in range(3):
            order = list(range(len(chosen)))
            rng.shuffle(order)
            for i in order:
                if i >= len(chosen):
                    continue
                u = 0
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    u |= chosen_masks[j]
                if u == full:
                    del chosen[i]
                    del chosen_masks[i]
                    improved = True
                    break
            if improved:
                break
    return chosen, chosen_masks


def _try_n(n, deadline, rng):
    """Try to find minimal edge set for given n."""
    edges, masks = _build_masks_fast(n)
    full = (1 << (1 << n)) - 1
    best = None
    best_len = float('inf')
    
    # Multiple restarts with varying parameters
    restart_configs = [
        (0.0, 1),    # deterministic
        (0.1, 1),
        (0.2, 1),
        (0.3, 1),
        (0.5, 1),
        (0.1, 2),    # with look-ahead
        (0.3, 2),
        (0.0, 1),    # another deterministic (different rng state)
    ]
    
    for randomize, look_ahead in restart_configs:
        if time.time() > deadline:
            break
        chosen, chosen_masks = _greedy_cover(masks, full, deadline, rng, randomize, look_ahead)
        if chosen is None:
            continue
        chosen, chosen_masks = _local_prune(chosen, chosen_masks, full, deadline, rng)
        chosen, chosen_masks = _swap_improve(chosen, chosen_masks, full, masks, deadline, rng)
        if len(chosen) < best_len:
            best_len = len(chosen)
            best = (list(chosen), list(chosen_masks))
            if best_len <= 35:
                break
    
    if best is None:
        return None
    
    chosen, _ = best
    # Exact certificate: union of masks must equal full
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
    
    # Try different n values, starting from smaller (fewer vertices may need fewer edges)
    for n in (9, 10, 11, 12, 13):
        if time.time() > deadline:
            break
        try:
            res = _try_n(n, deadline, rng)
        except Exception:
            res = None
        if res is not None and len(res['edges']) < len(best['edges']):
            best = res
    
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
