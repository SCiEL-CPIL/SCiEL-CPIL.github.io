"""Size-efficient sorting networks on 18 wires: search for minimal comparator count."""
import time
import numpy as np

N = 18
ALL_MASKS = None
ALL_COUNT = 1 << N
DEADLINE = 84.0  # safety margin under the 90s wall-clock limit


def _init_masks():
    global ALL_MASKS
    if ALL_MASKS is None:
        idx = np.arange(ALL_COUNT, dtype=np.uint32)
        ALL_MASKS = [(idx >> w) & 1 for w in range(N)]


def _apply_net(net, masks):
    """Apply comparator network to all 2^N Boolean inputs. Returns array of shape (N, ALL_COUNT)."""
    vals = np.stack(masks, axis=0)  # (N, ALL_COUNT)
    for a, b in net:
        va = vals[a]
        vb = vals[b]
        lo = np.minimum(va, vb)
        hi = np.maximum(va, vb)
        vals[a] = lo
        vals[b] = hi
    return vals


def _is_sorted(vals):
    """Check if the network sorts all Boolean inputs (i.e., output is a nondecreasing function of wire index)."""
    return np.all(vals[1:] >= vals[:-1])


def _count_inversions(vals):
    """Average number of inversions over all inputs (proxy for sorting progress)."""
    inv = 0
    for i in range(N):
        for j in range(i + 1, N):
            inv += np.count_nonzero(vals[i] > vals[j])
    return inv


def _bit_parity_ok(a, b, depth):
    """At depth d, comparator (a,b) must have a and b of opposite parity (i+j even)."""
    return ((a + b) + depth) % 2 == 0


def _build_candidates(vals, depth):
    """Return list of (score, (a,b)) candidates for next comparator, sorted by best-first."""
    cands = []
    cur_inv = _count_inversions(vals)
    for a in range(N):
        for b in range(a + 1, N):
            if not _bit_parity_ok(a, b, depth):
                continue
            # Apply (a,b) and measure inversion change
            va = vals[a].copy()
            vb = vals[b].copy()
            lo = np.minimum(va, vb)
            hi = np.maximum(va, vb)
            # Quick estimate: count how many pairs are fixed
            # Use a cheaper proxy: number of positions where va>vb (these get fixed)
            # and how many new inversions are created
            fixed = np.count_nonzero(va > vb)
            # Estimate: each fixed pair reduces inversions by 1 on average
            # This is a heuristic; we use -fixed as score (lower is better)
            cands.append((-fixed, (a, b)))
    cands.sort(key=lambda x: x[0])
    return cands


def _dfs_search(max_depth, start_time):
    """Iterative-deepening DFS for a sorting network of at most max_depth comparators."""
    _init_masks()
    masks = ALL_MASKS
    vals = np.stack(masks, axis=0)  # (N, ALL_COUNT)
    best = None

    def dfs(depth, max_d, net, vals_cur):
        nonlocal best
        if time.time() > DEADLINE:
            return
        if _is_sorted(vals_cur):
            if best is None or len(net) < len(best):
                best = list(net)
            return
        if depth >= max_d:
            return
        # Generate candidates
        cands = _build_candidates(vals_cur, depth)
        # Limit branching for speed: take top K candidates
        K = 12
        for _, (a, b) in cands[:K]:
            if time.time() > DEADLINE:
                return
            va = vals_cur[a].copy()
            vb = vals_cur[b].copy()
            lo = np.minimum(va, vb)
            hi = np.maximum(va, vb)
            vals_cur[a] = lo
            vals_cur[b] = hi
            net.append((a, b))
            dfs(depth + 1, max_d, net, vals_cur)
            net.pop()
            vals_cur[a] = va
            vals_cur[b] = vb
        # Also try a couple of random candidates for diversity
        import random
        for _ in range(2):
            if time.time() > DEADLINE:
                return
            a = random.randrange(N)
            b = random.randrange(a + 1, N)
            if not _bit_parity_ok(a, b, depth):
                continue
            va = vals_cur[a].copy()
            vb = vals_cur[b].copy()
            lo = np.minimum(va, vb)
            hi = np.maximum(va, vb)
            vals_cur[a] = lo
            vals_cur[b] = hi
            net.append((a, b))
            dfs(depth + 1, max_d, net, vals_cur)
            net.pop()
            vals_cur[a] = va
            vals_cur[b] = vb

    for d in range(1, max_depth + 1):
        if time.time() > DEADLINE:
            break
        dfs(0, d, [], np.stack(masks, axis=0))
        if best is not None:
            break

    return best


def construct():
    start = time.time()
    _init_masks()

    # Try iterative-deepening search for a small network
    max_search_depth = 80
    result = _dfs_search(max_search_depth, start)

    if result is not None:
        # Verify
        vals = _apply_net(result, ALL_MASKS)
        if _is_sorted(vals):
            return result

    # Fallback: odd-even transposition network (153 comparators)
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
