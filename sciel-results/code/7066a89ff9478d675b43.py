import time
import itertools
import numpy as np

_TIME_LIMIT = 110.0  # hard wall-clock budget (safety margin under 120s)


def _initial_seed(q, n):
    # Direct sum of the ternary tetracode (perfect radius-1 code of length 4)
    # with all words on the remaining coordinates -> 9 * q**(n-4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                 for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _base_indices(cw, q, n):
    p = np.array(cw, dtype=np.int64)
    return (p[:, None] * (q ** np.arange(n - 1, -1, -1, dtype=np.int64)))


def _distances_from_base(base, q, n):
    base = base.astype(np.int64)
    qpow = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    qpow = qpow.astype(np.int64)
    rem = base[:, None, :] - base[None, :, :]
    rem = np.mod(rem, q)
    rem = np.where(rem < 0, rem + q, rem)
    rem //= qpow
    d = (rem != 0).sum(axis=2)
    return d.astype(np.int32)


def _greedy_add(cw, q, n, radius, time_limit, start_time,
                initial_covered=None, init_d=None):
    qpow = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    qpow = qpow.astype(np.int64)
    qpow1 = qpow[1:]  # length n-1

    if initial_covered is None:
        initial_covered = np.zeros(q ** n, dtype=np.uint8)
    covered = initial_covered
    cw = list(cw)
    base = _base_indices(cw, q, n)
    d = _distances_from_base(base, q, n)
    dmin = d.min(axis=1)
    uncovered = dmin > radius
    n_un = int(uncovered.sum())
    n_total = int(len(covered))

    while n_un > 0 and (time.time() - start_time) < time_limit:
        # candidate base index for each uncovered word
        cand_base = np.nonzero(uncovered)[0]
        # distance of candidate from each existing codeword
        dc = np.mod(cand_base[:, None] - base[None, :, :], q)
        dc = np.where(dc < 0, dc + q, dc)
        dc //= qpow
        dcd = (dc != 0).sum(axis=2)
        dcd = dcd.astype(np.int32)
        # candidate is a valid new codeword iff it is >radius from every codeword
        valid = (dcd > radius).all(axis=1)
        if not valid.any():
            break
        # for each valid candidate, how many currently-uncovered words it covers
        cov = np.where((dcd <= radius) & uncovered[None, :], 1, 0).sum(axis=1)
        cov = np.where(valid, cov, -1)
        i = int(np.argmax(cov))
        if cov[i] <= 0:
            break
        b = int(cand_base[i])
        # mark covered
        covered[cand_base[valid]] = 1
        n_un -= int((cov[valid] > 0).sum())
        if n_un <= 0:
            break
        # update codeword bookkeeping
        cw.append((b // qpow).tolist())
        base = np.vstack([base, [b]])
        d = _distances_from_base(base, q, n)
        dmin = d.min(axis=1)
        uncovered = dmin > radius
        n_un = int(uncovered.sum())
    return cw, covered, d, n_un


def _local_search(cw, q, n, radius, time_limit, start_time,
                  init_covered, init_d, max_rounds=15):
    qpow = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    qpow = qpow.astype(np.int64)
    qpow1 = qpow[1:]
    covered = init_covered
    cw = list(cw)
    base = _base_indices(cw, q, n)
    d = init_d
    dmin = d.min(axis=1)
    uncovered = dmin > radius
    n_un = int(uncovered.sum())

    for _ in range(max_rounds):
        if n_un > 0:
            break
        if (time.time() - start_time) > time_limit:
            break
        m = len(cw)
        # for each codeword, count how many words it is the unique nearest coverer of
        # (i.e. removing it would leave them uncovered)
        # second-min distance over other codewords
        d_other = d.copy()
        for i in range(m):
            d_other[i, i] = 9999
        second_min = d_other.min(axis=1)
        # a word is uniquely covered by codeword i iff dmin==d[i] and second_min>radius
        # and d[i] is the min. Count per codeword.
        # unique(i) = number of words where d[i] == dmin and second_min > radius
        # build per-codeword unique count
        is_min = (d == dmin[:, None])
        is_unique = is_min & (second_min[:, None] > radius)
        unique_counts = is_unique.sum(axis=0)
        # find codeword with smallest unique count
        i = int(np.argmin(unique_counts))
        if unique_counts[i] > 0:
            break
        # codeword i is redundant -> remove it
        del cw[i]
        base = np.delete(base, i, axis=0)
        d = _distances_from_base(base, q, n)
        dmin = d.min(axis=1)
        uncovered = dmin > radius
        n_un = int(uncovered.sum())

    return cw, covered, d, n_un


def construct(q, n, radius):
    start_time = time.time()
    time_limit = _TIME_LIMIT

    # lower bound: cannot go below ceil(q**n / (1 + n*(q-1)))
    lb = (q ** n) // (1 + n * (q - 1))
    if (q ** n) % (1 + n * (q - 1)) > 0:
        lb += 1

    cw = _initial_seed(q, n)
    base = _base_indices(cw, q, n)
    d = _distances_from_base(base, q, n)
    dmin = d.min(axis=1)
    uncovered = dmin > radius
    n_un = int(uncovered.sum())

    # Phase 1: greedy add until covered or target reached or time out
    cw, covered, d, n_un = _greedy_add(
        cw, q, n, radius, time_limit, start_time,
        initial_covered=covered, init_d=d)

    # Phase 2: local search (redundancy removal) while still valid
    if n_un == 0:
        cw, covered, d, n_un = _local_search(
            cw, q, n, radius, time_limit, start_time,
            init_covered=covered, init_d=d)

    # safety: if still not fully covered, fall back to the always-valid seed
    if n_un > 0:
        cw = _initial_seed(q, n)

    # final validity guard
    base = _base_indices(cw, q, n)
    d = _distances_from_base(base, q, n)
    if d.min(axis=1).max() > radius:
        cw = _initial_seed(q, n)

    return cw
