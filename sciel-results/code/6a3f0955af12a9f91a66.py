import time
import random
import sys
import numpy as np

# ---------------- global scratch buffers (allocated once) ----------------
_MAXK = 64
_MAXROWS = 64
_buf = np.zeros((_MAXROWS, _MAXK), dtype=np.int64)   # candidate row being built
_used = np.zeros(1 << 16, dtype=np.uint8)           # used-difference bitset (max diff 65535)
_seen = np.zeros(1 << 16, dtype=np.uint8)           # seen-candidates bitset
_rng = np.random.default_rng(0)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 117.0

    k1 = k + 1
    # lower bound on scope: need n_rows * k*(k+1)/2 distinct positive differences in [1..scope]
    lower_bound = n_rows * k * (k + 1) // 2 - 1

    # 1) greedy baseline
    best = _greedy(n_rows, k)
    if not best:
        return []
    best_scope = max(max(r) for r in best)

    if best_scope <= lower_bound:
        return best

    # 2) fast local-search polish
    best, best_scope = _local_search(best, n_rows, k, deadline - 3.0)
    if best_scope <= lower_bound:
        return best

    # 3) random-restart backtracking at decreasing target scopes
    target = best_scope - 1
    attempt = 0
    while target >= lower_bound and time.time() < deadline - 4.0:
        found = False
        # more restarts the closer we are to the bound (harder)
        max_restarts = 2000
        for _ in range(max_restarts):
            if time.time() > deadline - 4.0:
                break
            sol = _backtrack(n_rows, k, target, deadline - 4.0, attempt)
            if sol is not None:
                s = max(max(r) for r in sol)
                if s < best_scope:
                    best, best_scope = sol, s
                    found = True
                    best, best_scope = _local_search(best, n_rows, k, deadline - 3.0)
                    if best_scope <= lower_bound:
                        return best
                    break
            attempt += 1
        if found:
            target = best_scope - 1
        else:
            # can't close this gap; try a slightly looser target to make progress
            target -= 1
            if target < lower_bound:
                break

    # final polish
    best, best_scope = _local_search(best, n_rows, k, deadline - 1.0)
    return best


def _greedy(n_rows, k):
    k1 = k + 1
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k1:
            x += 1
            nd = {x - a for a in row}
            if not (nd & used):
                row.append(x)
                used |= nd
        rows.append(row)
    return rows


def _backtrack(n_rows, k, max_scope, deadline, seed):
    """Build n_rows rows with all entries in [0, max_scope], distinct differences.
    Uses a global numpy bitset for used differences and a candidate buffer.
    Returns list of rows or None."""
    global _used, _seen, _buf, _rng
    rng = np.random.default_rng(seed * 2654435761 + 12345)
    _used[:] = 0
    k1 = k + 1
    rows = []

    def build_row(depth):
        # depth = number of entries already placed (row[0]=0 always)
        # place row[depth]
        global _used, _seen, _buf
        # candidate range: > row[depth-1], and leave room for remaining (k1-1-depth) entries
        prev = _buf[depth - 1]
        max_x = max_scope - (k1 - 1 - depth)
        if max_x <= prev:
            return False
        lo = prev + 1
        hi = max_x

        # compute for each candidate x whether it conflicts with already-placed entries
        # vectorized: for each x, differences x - row[i] for i in [0..depth-1]
        placed = _buf[:depth]                       # (depth,)
        xs = np.arange(lo, hi + 1)                 # (n,)
        # diffs shape (n, depth): xs[:,None] - placed[None,:]
        # a candidate is valid if none of its diffs is in _used
        # Build a boolean mask: for each (x, i) check used[x - row[i]]
        # differences range 1..max_scope (<=65535)
        diff_mat = xs[:, None] - placed[None, :]   # (n, depth)
        # gather used bits
        mask = _used[diff_mat]                     # (n, depth) uint8
        valid = ~mask.any(axis=1)                  # (n,) bool
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            return False
        cand = xs[idx]

        # MRV-ish: prefer smaller candidates, with a random spread
        # take a bounded candidate set
        if cand.size > 40:
            # smallest 20 + 20 random from the rest
            small = cand[:20]
            rest = cand[20:]
            pick = rng.choice(rest.size, size=min(20, rest.size), replace=False)
            extra = rest[pick]
            cand = np.concatenate([small, extra])
        order = rng.permutation(cand.size)
        cand = cand[order]

        for x in cand:
            x = int(x)
            # mark differences with placed entries
            placed = _buf[:depth]
            diffs = x - placed
            _used[diffs] = 1
            _buf[depth] = x
            if depth + 1 == k1:
                rows.append(_buf[:k1].tolist())
                _used[diffs] = 0
                return True
            if build_row(depth + 1):
                return True
            _used[diffs] = 0
            _buf[depth] = 0
        return False

    for _ in range(n_rows):
        _buf[0] = 0
        if not build_row(1):
            return None
    return rows


def _local_search(solution, n_rows, k, deadline):
    """Pull each element to its smallest feasible value; iterate to fixed point.
    Fast: rebuild used-set as a numpy bitset and vectorize per-position checks."""
    global _used
    k1 = k + 1
    best = [list(r) for r in solution]
    best_scope = max(max(r) for r in best)

    def used_set(rows):
        u = np.zeros(1 << 16, dtype=np.uint8)
        for r in rows:
            for i in range(k1):
                for j in range(i + 1, k1):
                    u[r[j] - r[i]] = 1
        return u

    def pull(rows):
        u = used_set(rows)
        changed = False
        for ri in range(n_rows):
            r = rows[ri]
            for pos in range(1, k1):
                cur = r[pos]
                # try values from r[pos-1]+1 up to cur
                for nv in range(r[pos - 1] + 1, cur + 1):
                    ok = True
                    # diffs with earlier
                    for i in range(pos):
                        d = nv - r[i]
                        if d <= 0 or u[d]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # diffs with later
                    for j in range(pos + 1, k1):
                        d = r[j] - nv
                        if d <= 0 or u[d]:
                            ok = False
                            break
                    if ok:
                        # remove old diffs of r[pos]
                        for i in range(pos):
                            u[r[pos] - r[i]] = 0
                        for j in range(pos + 1, k1):
                            u[r[j] - r[pos]] = 0
                        # add new
                        for i in range(pos):
                            u[nv - r[i]] = 1
                        for j in range(pos + 1, k1):
                            u[r[j] - nv] = 1
                        r[pos] = nv
                        changed = True
                        break
        return changed

    while time.time() < deadline:
        if not pull(best):
            break
    new_scope = max(max(r) for r in best)
    if new_scope < best_scope:
        return best, new_scope
    return [list(r) for r in solution], best_scope
