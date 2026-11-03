"""Second autocorrelation inequality: sparse Sidon-support design.

Maximizes ||h||_2^2 / (||h||_1 * ||h*h||_inf) over 50 nonnegative heights.
For a k-point support with equal nonzero heights the ratio equals 1/p where
p is the maximum number of repeated differences in the support.  A near-Sidon
support (all differences distinct, p=1) therefore yields ratio 1, which beats
the constant seed (ratio 1/sqrt(N) ~ 0.141).
"""
import time
import numpy as np

N = 50

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 3.0 - (time.time() - _START)


def _autoconv_peak(support):
    """Max number of ordered pairs (i, j) in support with i - j = c (c != 0)."""
    s = np.asarray(support, dtype=int)
    if s.size < 2:
        return 1
    diffs = s[:, None] - s[None, :]
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 1
    _, cnt = np.unique(nz, return_counts=True)
    return int(cnt.max())


def _greedy_sidon(seed_support, target_k, rng, max_iters):
    """Greedily extend seed_support to target_k points minimizing repeated diffs."""
    support = list(seed_support)
    if len(support) >= target_k:
        return support[:target_k]

    diff_count = {}
    for a in support:
        for b in support:
            if a != b:
                diff_count[a - b] = diff_count.get(a - b, 0) + 1

    for _ in range(max_iters):
        if len(support) >= target_k:
            break
        if _remaining() < 2.0:
            break
        # Candidate scores: how many new repeated differences it would create.
        best_idx, best_score = None, None
        for idx in range(N):
            if idx in support:
                continue
            score = 0
            for a in support:
                for d in (idx - a, a - idx):
                    score += diff_count.get(d, 0) + 1
            if best_score is None or score < best_score:
                best_score, best_idx = score, idx
                if score == 0:
                    break
        if best_idx is None:
            break
        support.append(best_idx)
        for a in support[:-1]:
            d1 = best_idx - a
            d2 = a - best_idx
            diff_count[d1] = diff_count.get(d1, 0) + 1
            diff_count[d2] = diff_count.get(d2, 0) + 1
    return support


def _ratio_from_support(support, k):
    if k <= 0:
        return 0.0
    p = _autoconv_peak(support)
    return 1.0 / float(p)


def _build():
    best_ratio = 0.0
    best_support = None

    # Deterministic baseline: greedy from index 0.
    sup0 = _greedy_sidon([0], N, None, 400)
    r0 = _ratio_from_support(sup0, N)
    if r0 > best_ratio:
        best_ratio, best_support = r0, sup0

    # A few seeded greedy attempts (cheap) to look for a full Sidon set.
    seeds = [
        [0, 1, 4, 10],
        [0, 1, 5, 12],
        [0, 2, 5, 9],
        [0, 1, 3, 7],
    ]
    rng = np.random.default_rng(20240517)
    for _ in range(8):
        seeds.append(sorted(set(int(x) for x in rng.integers(0, N, size=4))))

    for sd in seeds:
        if _remaining() < 1.5:
            break
        sd = [x for x in sorted(set(sd)) if 0 <= x < N]
        if not sd:
            continue
        sup = _greedy_sidon(sd, N, rng, 400)
        r = _ratio_from_support(sup, N)
        if r > best_ratio:
            best_ratio, best_support = r, sup
        if best_ratio >= 1.0:
            break  # full Sidon set found: ratio is optimal (can't exceed 1)

    return best_ratio, best_support


def construct():
    # Always keep a valid nonnegative vector; prefer the Sidon design if it wins.
    try:
        best_ratio, best_support = _build()
    except Exception:
        best_ratio, best_support = 0.0, None

    seed_ratio = 1.0 / float(N)  # constant vector ratio
    if best_support is None or best_ratio <= seed_ratio:
        return [1.0] * N

    heights = [0.0] * N
    for idx in best_support:
        heights[idx] = 1.0
    return heights
