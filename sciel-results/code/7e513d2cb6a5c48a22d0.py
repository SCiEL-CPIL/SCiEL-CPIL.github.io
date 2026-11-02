"""Deadline-guarded greedy covering-code builder for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools

import numpy as np

_DEADLINE = 150.0
_SAFETY = 12.0


def _now():
    return time.monotonic()


def _time_ok(deadline):
    return _now() < deadline


def _neighbors_table(q, n, radius):
    # Returns an array of shape (q**n, radius+1, n) with the coordinates of
    # Hamming balls of radius 0..radius around each word (as base-q index).
    total = q ** n
    table = np.empty((total, radius + 1, n), dtype=np.int32)
    coords = np.empty(n, dtype=np.int32)
    for idx in range(total):
        x = idx
        for i in range(n):
            coords[i] = x % q
            x //= q
        table[idx, 0] = coords
        for r in range(1, radius + 1):
            prev = table[idx, r - 1]
            for i in range(n):
                for d in range(q):
                    if d != coords[i]:
                        v = idx + (d - coords[i]) * (q ** i)
                        table[idx, r, i * q + d] = v
    return table


def _covered_mask(codes, table, total):
    # Boolean array over the whole space: word covered iff within radius of a code.
    covered = np.zeros(total, dtype=bool)
    for c in codes:
        for r in range(table.shape[1]):
            covered[table[c, r]] = True
    return covered


def _greedy_add(codes, covered, table, total, deadline):
    # Add one codeword that maximizes the number of newly-covered words.
    gains = np.zeros(total, dtype=np.int32)
    for c in codes:
        for r in range(table.shape[1]):
            gains[table[c, r]] += 1
    uncovered = total - covered.sum()
    best_idx = int(np.argmax(gains))
    best_gain = int(gains[best_idx])
    if best_gain <= 0:
        return False
    codes.append(best_idx)
    for r in range(table.shape[1]):
        covered[table[best_idx, r]] = True
    return True


def _local_improve(codes, table, total, deadline):
    # Try replacing one codeword with another that is no worse; also try dropping.
    improved = True
    while improved and _time_ok(deadline):
        improved = False
        covered = _covered_mask(codes, table, total)
        uncovered = total - covered.sum()
        n_codes = len(codes)
        for i in range(n_codes):
            if not _time_ok(deadline):
                break
            # Candidate removal
            covered_i = covered.copy()
            gains = np.zeros(total, dtype=np.int32)
            for c in codes:
                for r in range(table.shape[1]):
                    gains[table[c, r]] += 1
            for r in range(table.shape[1]):
                gains[table[codes[i], r]] -= 1
            if gains.sum() == 0:  # codeword i is redundant
                del codes[i]
                improved = True
                break
            # Try replacing i with the best alternative
            alt = int(np.argmax(gains))
            if gains[alt] > gains[codes[i]]:
                codes[i] = alt
                improved = True
                break
    return codes


def _to_lists(codes, q, n):
    out = []
    for c in codes:
        x = c
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        out.append(w)
    return out


def construct(q, n, radius):
    start = _now()
    deadline = start + _DEADLINE - _SAFETY
    total = q ** n

    # Trivial / small cases
    if n == 0:
        return [[]]
    if radius >= n:
        return [[0] * n]
    if radius == 0:
        return [list(itertools.product(range(q), repeat=n))[i] for i in range(total)]

    table = _neighbors_table(q, n, radius)

    # Lower bound on number of codewords needed (sphere-packing style).
    ball_size = sum(len(table[0, r]) for r in range(table.shape[1]))
    lower = int(np.ceil(total / ball_size))

    codes = []
    covered = np.zeros(total, dtype=bool)

    # Seed with a greedy start
    while _time_ok(deadline):
        if not _greedy_add(codes, covered, table, total, deadline):
            break
        if len(codes) >= lower and covered.all():
            break
        if len(codes) >= total:
            break

    # Local improvement pass
    codes = _local_improve(codes, table, total, deadline)

    return _to_lists(codes, q, n)
# EVOLVE-BLOCK-END
