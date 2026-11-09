# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_DEADLINE = 620.0  # safety margin under 630s kill
_START = time.time()


def _int_points_ball(R2):
    """All integer vectors in Z^11 with squared norm <= R2, sorted by norm."""
    R = int(np.sqrt(R2))
    lo = -R
    hi = R
    pts = []
    # iterate over 11 coordinates; prune by remaining budget
    coords = list(product(range(lo, hi + 1), repeat=11))
    # The above is huge for R>=2; instead generate recursively with pruning.
    pts = []
    def rec(dim, prefix, used):
        if used > R2:
            return
        if dim == 11:
            pts.append(tuple(prefix))
            return
        # bound on remaining dims: each can contribute up to R2 - used
        for v in range(-R, R + 1):
            nu = used + v * v
            if nu > R2:
                continue
            prefix.append(v)
            rec(dim + 1, prefix, nu)
            prefix.pop()
    rec(0, [], 0)
    # sort by squared norm
    pts.sort(key=lambda p: sum(x * x for x in p))
    return pts


def _min_pair_sqdist(accepted):
    """Minimum squared distance among accepted points (returns inf if <2)."""
    n = len(accepted)
    if n < 2:
        return float('inf')
    best = float('inf')
    for i in range(n):
        pi = accepted[i]
        for j in range(i + 1, n):
            pj = accepted[j]
            d2 = 0
            for a, b in zip(pi, pj):
                d = a - b
                d2 += d * d
            if d2 < best:
                best = d2
    return best


def _max_norm_sq(accepted):
    best = 0
    for p in accepted:
        n2 = sum(x * x for x in p)
        if n2 > best:
            best = n2
    return best


def kissing_number11() -> np.ndarray:
    """
    Greedy integer kissing configuration in dimension 11.

    Returns:
        points: np.ndarray of shape (k, 11) with int64 entries.
    """
    d = 11
    best_points = [tuple([1] + [0] * (d - 1)), tuple([-1] + [0] * (d - 1))]
    best_k = 2

    # Sweep increasing radius R; stop on deadline or when candidate pool is empty.
    for R in range(1, 40):
        if time.time() > _DEADLINE:
            break
        R2 = R * R
        try:
            candidates = _int_points_ball(R2)
        except Exception:
            break
        if len(candidates) < 2:
            continue

        # Greedy packing with min pair distance >= R2 + 1
        accepted = []
        target_min = R2 + 1
        for p in candidates:
            if time.time() > _DEADLINE:
                break
            ok = True
            for q in accepted:
                d2 = 0
                for a, b in zip(p, q):
                    diff = a - b
                    d2 += diff * diff
                    if d2 >= target_min:
                        break
                if d2 < target_min:
                    ok = False
                    break
            if ok:
                accepted.append(p)

        k = len(accepted)
        if k >= 2:
            # verify validity
            mn = _min_pair_sqdist(accepted)
            mx = _max_norm_sq(accepted)
            if mn >= mx and mn >= target_min:
                if k > best_k:
                    best_points = accepted
                    best_k = k

        # If candidate pool is small relative to R, larger R likely won't help much
        # but keep sweeping a bit more.

    arr = np.array(best_points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr


# EVOLVE-BLOCK-END
