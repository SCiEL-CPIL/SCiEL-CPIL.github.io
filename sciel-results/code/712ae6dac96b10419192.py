import time
import numpy as np

N = 18
MAX_COMPARATORS = 4096
WALL_LIMIT = 90.0
SAFETY_MARGIN = 10.0

# Precompute the full set of Boolean inputs as packed 64-bit lane words.
# Each of the 2**18 inputs is one 18-bit mask; lanes hold 3 masks per 64-bit word.
_LANES = 4  # 4 * 64 = 256 bits >= 18*14
_LANEMASK = (1 << 18) - 1
_ALL_INPUTS = np.arange(1 << 18, dtype=np.uint64)


def _verify(comp):
    """Return True iff the comparator list sorts every Boolean input."""
    if not comp:
        return N <= 1
    c = np.asarray(comp, dtype=np.int64)
    if c.size < 2:
        return False
    if np.any(c[:, 0] >= c[:, 1]) or np.any(c[:, 0] < 0) or np.any(c[:, 1] >= N):
        return False
    wires = np.zeros((1 << 18, N), dtype=np.uint64)
    for lane in range(_LANES):
        wires[:, lane * 18:(lane + 1) * 18] = np.ascontiguousarray(
            (_ALL_INPUTS >> (lane * 18)) & _LANEMASK
        )
    lo = np.minimum(wires[:, c[:, 0]], wires[:, c[:, 1]])
    hi = np.maximum(wires[:, c[:, 0]], wires[:, c[:, 1]])
    wires[:, c[:, 0]] = lo
    wires[:, c[:, 1]] = hi
    for k in range(1, N):
        if np.any(wires[:, k - 1] > wires[:, k]):
            return False
    return True


def _bitonic_merge(N, a, b, ascending):
    """Batcher bitonic merge for wires [a, b), returns list of (i, j)."""
    n = b - a
    if n <= 1:
        return []
    h = n // 2
    comps = _bitonic_merge(N, a, a + h, not ascending)
    comps += _bitonic_merge(N, a + h, b, ascending)
    comps += [(a + i, a + h + i) for i in range(h)]
    return comps


def _bitonic_sort(N):
    """Batcher bitonic sorting network for N wires."""
    comps = []
    size = 2
    while size < N:
        for i in range(0, N, size):
            comps += _bitonic_merge(N, i, min(i + size, N), True)
        size *= 2
    comps += _bitonic_merge(N, 0, N, True)
    return comps


def _greedy_remove(net):
    """Repeatedly remove the first removable comparator (targeted, end-biased)."""
    changed = True
    while changed:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            trial = net[:idx] + net[idx + 1:]
            if _verify(trial):
                net = trial
                changed = True
                break
    return net


def construct():
    start = time.time()
    deadline = start + WALL_LIMIT - SAFETY_MARGIN
    rng = np.random.default_rng(12345)

    # Valid starting points, ordered from small to large.
    candidates = [
        _bitonic_sort(N),
        [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)],
    ]

    best = None
    for cand in candidates:
        if len(cand) > MAX_COMPARATORS:
            continue
        if _verify(cand):
            best = list(cand)
            break
    if best is None:
        best = candidates[1]  # odd-even transposition is always valid

    def time_left():
        return deadline - time.time()

    # Phase 1: targeted greedy removal (measured to escape local minima).
    while time_left() > 3.0 and len(best) > 1:
        trial = _greedy_remove(list(best))
        if len(trial) < len(best):
            best = trial
        else:
            break

    # Phase 2: random comparator swaps (add/remove/replace) to escape plateaus.
    while time_left() > 2.0:
        net = list(best)
        m = len(net)
        op = rng.integers(0, 3)
        if op == 0 and m > 1:  # remove a random comparator
            k = int(rng.integers(0, m))
            net = net[:k] + net[k + 1:]
        elif op == 1:  # add a random comparator
            i, j = sorted(rng.choice(N, size=2, replace=False).tolist())
            net.append((i, j))
        else:  # replace a random comparator
            if m > 0:
                k = int(rng.integers(0, m))
                i, j = sorted(rng.choice(N, size=2, replace=False).tolist())
                net[k] = (i, j)
        if len(net) <= MAX_COMPARATORS and _verify(net):
            if len(net) < len(best):
                best = net

    return best
