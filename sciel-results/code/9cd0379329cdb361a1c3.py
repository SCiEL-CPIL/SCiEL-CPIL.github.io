"""
Combined approach: fast local search with incremental penalty tracking
to find minimum-color valid 4-AP colorings of [1..128].
"""
import time
import random

# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------
N = 128
_DEADLINE = 85.0  # safety margin before 90 s kill
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

# Build all non-constant 4-term APs (a, a+d, a+2d, a+3d), 0-indexed
_APS = []
_idx_to_aps = [[] for _ in range(N)]
for a in range(N):
    max_d = (N - 1 - a) // 3
    for d in range(1, max_d + 1):
        b, c, e = a + d, a + 2 * d, a + 3 * d
        ap = (a, b, c, e)
        ap_idx = len(_APS)
        _APS.append(ap)
        for idx in ap:
            _idx_to_aps[idx].append(ap_idx)

_NUM_APS = len(_APS)

# ---------------------------------------------------------------------------
# Fast "is this AP bad?" check (uses <= 2 distinct colors)
# ---------------------------------------------------------------------------
def _is_bad(c1, c2, c3, c4):
    if c1 == c2:
        if c3 == c4:
            return True
        return c1 == c3 or c1 == c4
    elif c1 == c3:
        return c2 == c4
    elif c1 == c4:
        return c2 == c3
    else:
        if c2 == c3:
            return c2 == c4
        elif c2 == c4:
            return c1 == c3
        elif c3 == c4:
            return c2 == c3
        return False

def _ap_bad_fast(colors, ap):
    return _is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])

# ---------------------------------------------------------------------------
# Penalty computation and local search
# ---------------------------------------------------------------------------
def _init_penalty(colors):
    """Return (penalty, ap_bad_list)."""
    ap_bad = [False] * _NUM_APS
    penalty = 0
    for i, ap in enumerate(_APS):
        if _ap_bad_fast(colors, ap):
            ap_bad[i] = True
            penalty += 1
    return penalty, ap_bad

def _local_search_k(colors, K, time_budget):
    """
    Local search: random restarts + hill-climbing on penalty.
    colors: list of length N with values in [0, K).
    Returns best valid coloring found (penalty 0) or None.
    """
    start = time.time()
    best_found = None
    rng = random.Random(hash(tuple(colors)) & 0xFFFFFFFF)

    for restart in range(200):
        if time.time() - start > time_budget or _time_left() < 3.0:
            break

        # Random init
        cur = [rng.randint(0, K - 1) for _ in range(N)]
        penalty, ap_bad = _init_penalty(cur)

        if penalty == 0:
            return cur

        # Hill-climb
        stuck = 0
        for step in range(5000):
            if time.time() - start > time_budget or _time_left() < 3.0:
                break

            # Pick a random position
            i = rng.randint(0, N - 1)
            old = cur[i]

            best_c = old
            best_pen = penalty

            # Try each color
            for c in range(K):
                if c == old:
                    continue
                cur[i] = c
                delta = 0
                for ap_idx in _idx_to_aps[i]:
                    was_bad = ap_bad[ap_idx]
                    now_bad = _ap_bad_fast(cur, _APS[ap_idx])
                    if was_bad and not now_bad:
                        delta -= 1
                    elif not was_bad and now_bad:
                        delta += 1
                cur[i] = old
                new_pen = penalty + delta
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_c = c

            if best_c == old:
                stuck += 1
                if stuck > 300:
                    break
                continue

            # Apply best_c
            cur[i] = best_c
            for ap_idx in _idx_to_aps[i]:
                was_bad = ap_bad[ap_idx]
                now_bad = _ap_bad_fast(cur, _APS[ap_idx])
                if was_bad and not now_bad:
                    penalty -= 1
                    ap_bad[ap_idx] = False
                elif not was_bad and now_bad:
                    penalty += 1
                    ap_bad[ap_idx] = True
            stuck = 0

            if penalty == 0:
                return cur

    return None

def _reduce_colors(colors, K, time_budget):
    """
    Given a valid coloring using <= K colors, try to reduce to fewer colors.
    Returns (reduced_colors, num_colors) or (None, K) if no reduction.
    """
    start = time.time()
    cur = list(colors)
    used = sorted(set(cur))
    nc = len(used)
    rng = random.Random(12345)

    # Remap to 0..nc-1
    remap = {c: i for i, c in enumerate(used)}
    cur = [remap[c] for c in cur]
    nc = len(used)

    # Try to merge color classes
    for target_nc in range(nc - 1, 2, -1):
        if time.time() - start > time_budget or _time_left() < 3.0:
            break
        # Try merging each pair
        for c1 in range(target_nc + 1):
            for c2 in range(c1 + 1, target_nc + 1):
                if time.time() - start > time_budget or _time_left() < 3.0:
                    break
                # Merge c2 into c1
                trial = [c1 if x == c2 else x for x in cur]
                # Check validity
                penalty, _ = _init_penalty(trial)
                if penalty == 0:
                    cur = trial
                    # Remap to 0..target_nc-1
                    used_new = sorted(set(cur))
                    remap2 = {c: i for i, c in enumerate(used_new)}
                    cur = [remap2[c] for c in cur]
                    nc = target_nc
                    break
            if nc == target_nc:
                break
        if nc == target_nc:
            break

    return cur, nc

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    global _START
    _START = time.time()

    best_colors = None
    best_nc = N + 1

    # Phase 1: find a valid coloring with small K via local search
    for K in range(3, 8):
        if _time_left() < 5.0:
            break
        budget = min(10.0, _time_left() * 0.3)
        result = _local_search_k([0] * N, K, budget)
        if result is not None:
            nc = len(set(result))
            if nc < best_nc:
                best_nc = nc
                best_colors = result
            break  # found valid with K colors; try to reduce

    # Fallback: if no valid coloring found yet, do a greedy construction
    if best_colors is None:
        if _time_left() < 2.0:
            # Emergency: return a trivial (invalid) coloring to avoid crash
            return list(range(N))
        # Greedy: assign smallest color that doesn't create a violation
        colors = [-1] * N
        for i in range(N):
            for c in range(6):
                colors[i] = c
                ok = True
                for ap_idx in _idx_to_aps[i]:
                    ap = _APS[ap_idx]
                    if -1 in [colors[j] for j in ap]:
                        continue
                    if _ap_bad_fast(colors, ap):
                        ok = False
                        break
                if ok:
                    break
            if colors[i] == -1:
                colors[i] = 0
        # Verify
        penalty, _ = _init_penalty(colors)
        if penalty == 0:
            best_colors = colors
            best_nc = len(set(colors))
        else:
            # Try local search to fix
            K = len(set(colors))
            result = _local_search_k(colors, K, min(15.0, _time_left() * 0.5))
            if result is not None:
                best_colors = result
                best_nc = len(set(result))
            else:
                best_colors = colors
                best_nc = len(set(colors))

    # Phase 2: try to reduce colors
    if _time_left() > 5.0:
        reduced, nc = _reduce_colors(best_colors, best_nc, min(20.0, _time_left() * 0.4))
        if reduced is not None and nc < best_nc:
            best_colors = reduced
            best_nc = nc

    # Phase 3: final local search polish to ensure validity
    if _time_left() > 3.0:
        penalty, _ = _init_penalty(best_colors)
        if penalty > 0:
            K = max(best_nc, 3)
            result = _local_search_k(best_colors, K, min(10.0, _time_left() * 0.5))
            if result is not None:
                best_colors = result
                best_nc = len(set(best_colors))

    # Ensure labels are in 0..127
    used = sorted(set(best_colors))
    remap = {c: i for i, c in enumerate(used)}
    best_colors = [remap[c] for c in best_colors]

    return best_colors
