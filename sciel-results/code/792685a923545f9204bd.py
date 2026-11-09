# EVOLVE-BLOCK-START
import time

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVS)


def _cands_in_window(d, L, W):
    """All multiples of d in [L, L+W], sorted."""
    k = (L + d - 1) // d
    vals = []
    v = k * d
    hi = L + W
    while v <= hi:
        vals.append(v)
        v += d
    return vals


def _solve_window(L, W):
    """Backtrack: assign distinct multiples to each divisor within [L, L+W].
    Returns list of chosen values in divisor order, or None."""
    # Precompute candidate lists; sort divisors by fewest candidates (MRV)
    cand_lists = [_cands_in_window(d, L, W) for d in _DIVS]
    order = list(range(_N))
    order.sort(key=lambda i: len(cand_lists[i]))

    used = set()
    assign = [None] * _N

    def bt(idx):
        if idx == _N:
            return True
        i = order[idx]
        for v in cand_lists[i]:
            if v not in used:
                used.add(v)
                assign[i] = v
                if bt(idx + 1):
                    return True
                used.discard(v)
                assign[i] = None
        return False

    if bt(0):
        return assign
    return None


def construct():
    deadline = time.time() + 85.0
    best_val = float('inf')
    best_assign = None

    # Try increasing window widths; for each, try many random L positions.
    # A smaller W that works gives a better (lower) raw value.
    for W in range(1, 300):
        if time.time() > deadline:
            break
        # Number of L tries scales with W (larger windows are easier to satisfy)
        # but cap total work. Try random L in [1, 1000000].
        import random
        rng = random.Random(12345 + W)
        # Try a few deterministic Ls plus random
        max_tries = 200
        found = False
        for t in range(max_tries):
            if time.time() > deadline:
                break
            L = rng.randint(1, 1000000 - W)
            assign = _solve_window(L, W)
            if assign is not None:
                # Valid: all distinct (backtrack enforces), all in window, all divisible
                best_val = W
                best_assign = assign[:]
                found = True
                break
        if found:
            # Found a solution at width W; try one more width to see if smaller exists?
            # No—W is increasing, so first W found is the smallest width tried.
            # But a smaller width might have been missed due to random L sampling.
            # Do a quick denser pass on W-1 if W>1.
            if W > 1:
                prev = W - 1
                rng2 = random.Random(99999 + prev)
                for t in range(400):
                    if time.time() > deadline:
                        break
                    L = rng2.randint(1, 1000000 - prev)
                    assign = _solve_window(L, prev)
                    if assign is not None:
                        best_val = prev
                        best_assign = assign[:]
                        break
            break

    if best_assign is None:
        # Fallback: return divisors themselves (always valid, distinct, <= 1e6)
        best_assign = _DIVS[:]

    # Ensure Python ints and a list
    return [int(v) for v in best_assign]
# EVOLVE-BLOCK-END
