import time
import random

_DEADLINE = 90.0
_MARGIN = 5.0


def _is_div_incomp(sums):
    """True iff no two distinct values in sums divide one another.

    sums: list/set of ints (all positive). O(k^2) with k = len(sums).
    """
    arr = sorted(sums)
    m = len(arr)
    for i in range(m):
        a = arr[i]
        if a <= 0:
            return False
        for j in range(i + 1, m):
            if arr[j] % a == 0:
                return False
    return True


def _can_add(arr, x):
    """Can x be added to arr while preserving the invariant?

    Invariant maintained: no two distinct nonempty subset sums divide one
    another, and all subset sums are unique. Adding x is safe iff:
      - x not already in arr
      - every new sum (x and s+x for existing s) is unique
      - no new sum divides / is divisible by any existing sum
    """
    if x in arr:
        return False
    sums = set()
    n = len(arr)
    for r in range(1, n + 1):
        acc = 0
        # build sums incrementally via bitmasks is fine for small n
    # simpler: bitmask
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        sums.add(s)
    new_sums = {x}
    for s in sums:
        new_sums.add(s + x)
    # uniqueness
    if len(new_sums) != 1 + len(sums):
        return False
    # divisibility between new and old
    for ns in new_sums:
        for os in sums:
            if ns != os and (ns % os == 0 or os % ns == 0):
                return False
    return True


def _greedy(arr, candidates, rng):
    """Incrementally grow arr by greedily adding compatible candidates."""
    for x in candidates:
        if _can_add(arr, x):
            arr.append(x)
    return arr


def _try_improve(arr, candidates, rng):
    """Attempt local swaps/adds to grow arr. Returns improved arr or None."""
    best = list(arr)
    # try adding any unused candidate
    used = set(best)
    for x in candidates:
        if x in used:
            continue
        if _can_add(best, x):
            best.append(x)
    # try 1-for-1 swaps to enable more additions
    improved = True
    while improved:
        improved = False
        cur = list(best)
        for i, a in enumerate(cur):
            for x in candidates:
                if x in used:
                    continue
                trial = [v for j, v in enumerate(cur) if j != i]
                trial.append(x)
                if _can_add(trial, x) or True:  # placeholder
                    pass
        # (swap logic kept simple: skip to stay fast)
    return best


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN

    # candidate pool: larger values are easier to keep divisibility-free
    pool = list(range(128, 257))
    rng = random.Random(12345)

    best = []

    # --- Phase 1: incremental greedy with many random restarts ---
    for restart in range(400):
        if time.time() > deadline:
            break
        order = pool[:]
        rng.shuffle(order)
        # bias toward larger values
        order.sort(reverse=True)
        cur = []
        for x in order:
            if _can_add(cur, x):
                cur.append(x)
        if len(cur) > len(best):
            best = cur

    # --- Phase 2: local improvement on the best set ---
    if time.time() <= deadline:
        cur = list(best)
        used = set(cur)
        # add any compatible pool element
        for x in pool:
            if x in used:
                continue
            if _can_add(cur, x):
                cur.append(x)
                used.add(x)
        if len(cur) > len(best):
            best = cur

    # --- Phase 3: randomized hill-climbing (add/remove) ---
    if time.time() <= deadline:
        cur = list(best)
        for _ in range(20000):
            if time.time() > deadline:
                break
            used = set(cur)
            # try to add a random unused candidate
            cands = [x for x in pool if x not in used]
            if cands:
                x = rng.choice(cands)
                if _can_add(cur, x):
                    cur.append(x)
            # occasionally remove a random element to allow re-growth
            elif len(cur) > 1 and rng.random() < 0.1:
                cur.pop(rng.randrange(len(cur)))
            if len(cur) > len(best):
                best = cur

    # final safety: ensure distinct, in range, plain ints
    best = [int(x) for x in dict.fromkeys(best)]
    best = [x for x in best if 1 <= x <= 256]
    return best
