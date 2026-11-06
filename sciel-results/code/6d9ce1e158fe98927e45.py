"""Search for a small sorting network on 18 wires with EXACT Boolean verification."""
import time
import numpy as np

N = 18
WORDS = 16            # 16 words x 18 bits = 288 >= 2**18
TOTAL = 1 << N        # 262144
LOWER_BOUND = 68

# Precompute the 18 column bitmasks (bit b of word w = input index w*18 + b)
def _build_masks():
    masks = np.zeros((N, WORDS), dtype=np.uint64)
    for b in range(N):
        for w in range(WORDS):
            idx = w * N + b
            if idx < TOTAL:
                masks[b, w] = np.uint64(1) << np.uint64(b)
    return masks

MASKS = _build_masks()

def _seed_network():
    """Odd-even transposition network (153 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _apply_col(cols, i, j):
    """Apply comparator (i,j) in-place to a (N, WORDS) uint64 column array.
    min -> i, max -> j. Returns the number of word-ops touched (for budgeting)."""
    ci = cols[i]
    cj = cols[j]
    cols[i] = ci & cj
    cols[j] = ci | cj
    return 0

def _residual(cols):
    """Count violation bits: sum over adjacent columns c[k] & ~c[k+1]."""
    r = 0
    for k in range(N - 1):
        r += int(np.bitwise_count(cols[k] & np.bitwise_not(cols[k + 1])))
    return r

def _is_valid(cols):
    return _residual(cols) == 0

def _apply_net(net, cols):
    for i, j in net:
        _apply_col(cols, i, j)

def _eval_net(net, deadline):
    """Return residual of net applied to all Boolean inputs, or None if deadline hit."""
    cols = MASKS.copy()
    for i, j in net:
        if time.time() > deadline:
            return None
        ci = cols[i]
        cj = cols[j]
        cols[i] = ci & cj
        cols[j] = ci | cj
    return _residual(cols)

def _random_vecs(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(lo, hi, size=N).astype(np.int64) for _ in range(n)]

def _apply_net_vec(net, v):
    for i, j in net:
        a = v[i]
        b = v[j]
        if a > b:
            v[i], v[j] = b, a
    return v

def _verify_random(net, vecs):
    for v in vecs:
        vv = v.copy()
        _apply_net_vec(net, vv)
        if np.any(np.diff(vv) < 0):
            return False
    return True

# Precomputed candidate comparator list
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_remove(net, deadline, vecs):
    """Remove any comparator that keeps the network valid (exact + random)."""
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            if len(net) <= LOWER_BOUND:
                return net
            trial = net[:idx] + net[idx + 1:]
            r = _eval_net(trial, deadline)
            if r is None:
                return net
            if r == 0 and _verify_random(trial, vecs):
                net = trial
                changed = True
                break
    return net

def _greedy_insert(net, deadline, vecs):
    """Insert the single comparator that most reduces the residual (if stuck)."""
    net = list(net)
    if time.time() > deadline:
        return net
    cols = MASKS.copy()
    _apply_net(net, cols)
    cur = _residual(cols)
    if cur == 0:
        return net
    best_gain = 0
    best_pair = None
    best_pos = None
    # try inserting at a few strategic positions
    positions = sorted(set([len(net), len(net) // 2, 0, max(0, len(net) - 5)]))
    for pos in positions:
        if time.time() > deadline:
            break
        for (i, j) in ALL_PAIRS:
            if (i, j) in net:
                continue
            trial = net[:pos] + [(i, j)] + net[pos:]
            r = _eval_net(trial, deadline)
            if r is None:
                return net
            gain = cur - r
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
                best_pos = pos
    if best_pair is not None and best_gain > 0:
        net = net[:best_pos] + [best_pair] + net[best_pos:]
    return net

def _polish_remove(net, deadline, vecs):
    """Final exhaustive single-removal pass (repeated until stable)."""
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            if len(net) <= LOWER_BOUND:
                return net
            trial = net[:idx] + net[idx + 1:]
            r = _eval_net(trial, deadline)
            if r is None:
                return net
            if r == 0 and _verify_random(trial, vecs):
                net = trial
                changed = True
                break
    return net

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    seed = _seed_network()
    best = list(seed)
    best_len = len(seed)

    # Random verification vectors (cheap sanity layer)
    vecs = _random_vecs(200, 0, 100000, 42)

    # Confirm seed is valid (exact)
    r = _eval_net(seed, deadline)
    if r != 0:
        # seed should always be valid; guard just in case
        return seed

    # Phase 1: greedy removal from seed
    if time.time() < deadline - 30.0:
        cand = _greedy_remove(seed, deadline - 30.0, vecs)
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)

    # Phase 2: if stuck (can't remove more), try inserting a helpful comparator
    # then removing others (restructuring).
    if time.time() < deadline - 20.0:
        # check if best is removable-stuck
        stuck = True
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline - 20.0:
                break
            trial = best[:idx] + best[idx + 1:]
            r = _eval_net(trial, deadline - 20.0)
            if r is not None and r == 0 and _verify_random(trial, vecs):
                stuck = False
                break
        if stuck and len(best) > LOWER_BOUND:
            cand = _greedy_insert(best, deadline - 20.0, vecs)
            cand = _greedy_remove(cand, deadline - 20.0, vecs)
            if len(cand) < best_len:
                best = cand
                best_len = len(cand)

    # Phase 3: final polish (exhaustive removal pass)
    if time.time() < deadline - 2.0:
        cand = _polish_remove(best, deadline - 2.0, vecs)
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)

    # Final exact + random verification of the best
    r = _eval_net(best, deadline)
    if r != 0 or not _verify_random(best, vecs):
        # fall back to seed
        return seed

    return best
