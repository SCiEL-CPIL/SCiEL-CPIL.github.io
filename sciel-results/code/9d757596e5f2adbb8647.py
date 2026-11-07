import time
import random


# ---- incremental (fast) verification state ----
# dp[k] is a bitset: bit s is set iff some k-subset sums to s.
# The rule is violated exactly when two DIFFERENT cardinalities share a sum.

class Verifier:
    __slots__ = ("n", "dp", "acc", "valid")

    def __init__(self, n):
        self.n = n
        self.dp = [0] * (n + 1)
        self.dp[0] = 1
        self.acc = 1          # union of all dp[k] so far
        self.valid = True     # no cross-cardinality collision yet

    def add(self, x):
        if not self.valid:
            return False
        n = self.n
        prev = self.dp
        ndp = list(prev)
        for k in range(n, 0, -1):
            ndp[k] = prev[k] | (prev[k - 1] << x)
        acc = self.acc
        for k in range(n + 1):
            if ndp[k] & acc:
                # this k's sums (now including new subsets) collide with
                # some other cardinality's sums -> rule broken
                self.valid = False
                return False
            acc |= ndp[k]
        self.dp = ndp
        self.acc = acc
        return True


def _check(arr):
    """Independent full verification (used before returning)."""
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_add(arr, used, ver, v):
    if v in used:
        return False
    arr.append(v)
    used.add(v)
    ok = ver.add(v)
    if not ok:
        arr.pop()
        used.discard(v)
    return ok


def _greedy(deadline, rng, order=None):
    """Deterministic/rule-based build: largest-valid-first, small gap."""
    cur = []
    used = set()
    ver = Verifier(60)
    if order is None:
        order = list(range(256, 0, -1))
        if rng is not None:
            rng.shuffle(order)

    added = True
    while added and len(cur) < 60 and time.monotonic() < deadline:
        added = False
        for v in order:
            if _try_add(cur, used, ver, v):
                added = True
                break

    # refine: try inserting mid-range values to grow further
    extra = list(range(1, 257))
    if rng is not None:
        rng.shuffle(extra)
    for v in extra:
        if time.monotonic() > deadline:
            break
        if _try_add(cur, used, ver, v):
            pass
    return cur


def _local_search(arr, deadline):
    """Try swapping elements in arr to see if we can add more elements."""
    if len(arr) >= 60:
        return arr

    cur = list(arr)
    used = set(cur)
    ver = Verifier(60)
    for v in cur:
        ver.add(v)

    # --- Pass A: 1-for-2 (remove one, add two) ---
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            cur.pop(i)
            used.discard(removed)

            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)

            added_count = 0
            for v in range(256, 0, -1):
                if v not in used and _try_add(cur, used, ver2, v):
                    added_count += 1
                    if added_count == 2:
                        break
                if time.monotonic() > deadline:
                    break

            if added_count >= 2:
                improved = True
            else:
                # Revert: pop added elements, restore removed
                for _ in range(added_count):
                    cur.pop()
                cur.insert(i, removed)
                used.add(removed)

    # --- Pass B: 1-for-1 swap (replace one element to unblock additions) ---
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            for new in range(256, 0, -1):
                if new == old or new in used:
                    continue
                # Try swap
                cur[i] = new
                used.discard(old)
                used.add(new)
                ver2 = Verifier(60)
                ok = True
                for v in cur:
                    if not ver2.add(v):
                        ok = False
                        break
                if not ok:
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
                    continue
                # After swap, try to add one more element
                added = False
                for v in range(256, 0, -1):
                    if v not in used and _try_add(cur, used, ver2, v):
                        added = True
                        break
                if added:
                    improved = True
                    break  # restart outer loop with new state
                else:
                    # revert swap
                    cur[i] = old
                    used.discard(new)
                    used.add(old)
        if improved:
            break  # restart while with updated cur

    # --- Pass C: remove-then-re-add polish ---
    # For each element, try removing it and re-adding a different one,
    # then attempt further additions.
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            cur.pop(i)
            used.discard(old)
            ver2 = Verifier(60)
            for v in cur:
                ver2.add(v)
            # Try re-adding a different element at this position
            found = False
            for new in range(256, 0, -1):
                if new == old or new in used:
                    continue
                if _try_add(cur, used, ver2, new):
                    found = True
                    break
            if found:
                improved = True
                break
            else:
                # Restore
                cur.insert(i, old)
                used.add(old)

    return cur


def construct():
    t0 = time.monotonic()
    deadline = t0 + 82.0  # explicit safety margin under 90s wall-clock kill
    best = None
    best_len = 0

    def consider(arr):
        nonlocal best, best_len
        if _check(arr) and len(arr) > best_len:
            best = list(arr)
            best_len = len(arr)
            return True
        return False

    # 1) deterministic greedy: largest-valid-first (structure-derived answer)
    g0 = _greedy(deadline, None)
    consider(g0)
    
    # 2) local search on the greedy result
    if best is not None:
        ls_result = _local_search(best, deadline)
        consider(ls_result)

    # 3) structured restarts: random order, keep global best
    rng = random.Random(2024)
    for _ in range(600):
        if time.monotonic() > deadline:
            break
        g = _greedy(deadline, rng)
        if len(g) > best_len:
            consider(g)
            # Apply local search to the new best
            if best is not None:
                ls_result = _local_search(best, deadline)
                consider(ls_result)
        if best_len >= 60:
            break

    if best is None:
        best = list(range(226, 257))
        if not _check(best):  # shouldn't happen
            best = list(range(1, 31))
    best = [int(v) for v in best]
    return best
