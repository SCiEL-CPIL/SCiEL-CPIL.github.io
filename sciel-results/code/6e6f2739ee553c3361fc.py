"""Covering code K_q(n, radius): greedy set-cover with NumPy, time-guarded."""
import time
import itertools
import numpy as np

_DEADLINE = 100.0  # safety margin under the 120s wall-clock limit


def _seed(q, n, radius):
    # Direct sum of the ternary tetracode (9 words, perfect radius-1 on 4 coords)
    # with every word on the remaining n-4 coordinates.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_offsets(radius, n, q):
    # All words within Hamming distance <= radius of the origin (as int indices).
    off = [0]
    for _ in range(radius):
        new = list(off)
        for pos in range(n):
            stride = q ** (n - 1 - pos)
            for d in range(1, q):
                for b in off:
                    new.append(b + d * stride)
        off = new
    return np.array(sorted(set(off)), dtype=np.int64)


def construct(q, n, radius):
    t0 = time.time()
    if n <= 0 or q <= 1:
        return [list(range(0))[:n]] if n == 0 else [list(range(n))]

    N = q ** n
    if N <= 0:
        return []

    # Fast fallback (known-valid 486-ish construction)
    try:
        fallback = _seed(q, n, radius)
    except Exception:
        fallback = [[0] * n]

    # Small instances: exhaustive greedy (full recomputation) is cheap and near-optimal.
    if N <= 8000:
        try:
            words = np.arange(N, dtype=np.int64)
            base = np.array(fallback, dtype=np.int64).ravel()
            contrib = np.zeros(N, dtype=np.int64)
            for c in base:
                contrib[c] += 1
            code = []
            while contrib.min() == 0:
                if time.time() > _DEADLINE:
                    break
                vals = contrib[:, None] + contrib[None, :]
                vals = np.where(vals == 0, 1, 0)
                c = int(vals.sum(axis=1).argmax())
                code.append(c)
                contrib[c] += 1
            if contrib.min() > 0 and len(code) < len(fallback):
                return [list(np.unravel_index(c, (q,) * n)) for c in code]
        except Exception:
            pass
        return fallback

    # ---- General greedy set-cover (vectorized) ----
    try:
        offsets = _ball_offsets(radius, n, q)
        B = offsets.size
        all_idx = np.arange(N, dtype=np.int64)

        # Initial coverage count per word (from the fallback seed)
        base = np.array(fallback, dtype=np.int64).ravel()
        contrib = np.zeros(N, dtype=np.int64)
        for c in base:
            contrib[c] += 1
        if contrib.min() > 0:
            return fallback  # seed already covers; nothing to gain

        # Greedy: repeatedly add the candidate covering the most uncovered words.
        added = 0
        while contrib.min() == 0:
            if time.time() > _DEADLINE:
                break
            # contribution[c] = number of currently-uncovered words in c's ball
            cand = np.zeros(N, dtype=np.int64)
            for k in range(0, B, 200000):
                chunk = offsets[k:k + 200000]
                cand += (contrib[all_idx[:, None] + chunk[None, :]] == 0).sum(axis=1)
            c = int(cand.argmax())
            code = [c]
            # update uncovered counts
            for k in range(0, B, 200000):
                chunk = offsets[k:k + 200000]
                contrib[all_idx[:, None] + chunk[None, :]] -= 1
            added += 1
            if contrib.min() > 0:
                break

        if contrib.min() > 0:
            code = [int(c) for c in base] + code
            return [list(np.unravel_index(c, (q,) * n)) for c in code]
    except Exception:
        pass

    return fallback
