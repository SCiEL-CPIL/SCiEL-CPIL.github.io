"""Difference triangle set (n_rows, k) — fresh design.

Strategy: each row is a Golomb ruler of k marks (k+1 integers, first = 0).
All positive intra-row differences must be globally distinct. We minimise the
scope = max entry. We use simulated annealing with a fast bitset conflict
check (per-row difference bitmasks + a global OR), multiple initialisations
(random / greedy / perturbed-greedy) and a hard wall-clock deadline guard.
"""
import time
import random

_DEADLINE = 145.0          # wall-clock guard (evaluator kills at 150s)
_MAX_SCOPE = 300           # cap on the difference universe we track


def _diff_mask(row):
    """Bitmask of positive differences of a strictly-increasing row (row[0]==0)."""
    m = 0
    L = len(row)
    for i in range(L):
        ai = row[i]
        for j in range(i + 1, L):
            d = row[j] - ai
            m |= 1 << d
    return m


def _row_scope(row):
    return row[-1]


def _greedy_row(used, k, start=1):
    """Smallest next mark whose new differences are all unused (Golomb step)."""
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        new = 0
        for a in row:
            new |= 1 << (x - a)
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _random_row(k, scope, rng):
    """Randomly pick k increasing marks within [1, scope]."""
    s = rng.sample(range(1, scope + 1), k)
    s.sort()
    return [0] + s


def _initialise(n_rows, k, mode, rng):
    rows = []
    used = 0
    if mode == "greedy":
        for _ in range(n_rows):
            row = _greedy_row(used, k)
            rows.append(row)
            used |= _diff_mask(row)
    elif mode == "random":
        for _ in range(n_rows):
            row = _random_row(k, _MAX_SCOPE, rng)
            rows.append(row)
            used |= _diff_mask(row)
    else:  # perturbed-greedy
        for _ in range(n_rows):
            row = _greedy_row(used, k)
            if rng.random() < 0.6:
                # shift a random interior mark up by a small random amount
                idx = rng.randrange(1, k)
                shift = rng.randint(1, 4)
                row[idx] += shift
                # keep strictly increasing
                for j in range(idx + 1, k + 1):
                    if row[j] <= row[j - 1]:
                        row[j] = row[j - 1] + 1
            rows.append(row)
            used |= _diff_mask(row)
    return rows


def _eval(rows, k):
    """Return (valid, scope, conflict_count)."""
    masks = [_diff_mask(r) for r in rows]
    g = 0
    for m in masks:
        g |= m
    # number of bits that appear in more than one row
    conflict = 0
    for m in masks:
        conflict += (m & (g ^ m)).bit_count()
    # each double-counted bit is counted twice -> divide by 2
    conflict //= 2
    scope = max(r[-1] for r in rows)
    valid = (conflict == 0)
    return valid, scope, conflict


def _repair(rows, k, rng):
    """Rebuild rows greedily, reusing a good prefix, to cut conflicts/scope."""
    used = 0
    new_rows = []
    for r in rows:
        # greedy from current used, but bias the start to be near r's last mark
        row = _greedy_row(used, k, start=max(1, r[-1] // 2))
        new_rows.append(row)
        used |= _diff_mask(row)
    return new_rows


def _anneal(rows, k, rng, deadline):
    """Simulated annealing over row perturbations."""
    valid, cur_scope, cur_conf = _eval(rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = cur_scope
    if valid:
        best_scope = min(best_scope, cur_scope)

    # energy = scope + big * conflict
    BIG = 1000
    T = max(1.0, float(best_scope))
    T_min = 0.1
    cooling = 0.985

    def energy(rws):
        v, s, c = _eval(rws, k)
        return s + BIG * c, v, s, c

    e_cur, v_cur, s_cur, c_cur = energy(rows)
    e_best = e_cur

    while time.time() < deadline:
        # pick a row to perturb
        ri = rng.randrange(len(rows))
        old = rows[ri][:]
        r = rows[ri]
        op = rng.random()
        if op < 0.45:
            # move one interior mark up or down
            idx = rng.randrange(1, k + 1)
            delta = rng.randint(-3, 3)
            if delta != 0:
                r[idx] += delta
        elif op < 0.7:
            # swap two interior marks then re-sort
            i, j = rng.sample(range(1, k + 1), 2)
            r[i], r[j] = r[j], r[i]
            tail = sorted(r[1:])
            r[1:] = tail
        elif op < 0.85:
            # replace a random mark with a fresh random value in range
            idx = rng.randrange(1, k + 1)
            lo = r[idx - 1] + 1
            hi = r[idx + 1] - 1 if idx < k + 1 else best_scope + 5
            if hi >= lo:
                r[idx] = rng.randint(lo, hi)
        else:
            # local greedy repair of this single row against global used-minus-row
            pass

        # enforce strictly increasing, fix violations
        for j in range(1, k + 1):
            if r[j] <= r[j - 1]:
                r[j] = r[j - 1] + 1

        e_new, v_new, s_new, c_new = energy(rows)
        if e_new <= e_cur or rng.random() < pow(2.718281828, -(e_new - e_cur) / max(T, 1e-9)):
            e_cur, v_cur, s_cur, c_cur = e_new, v_new, s_new, c_new
            if e_new < e_best:
                e_best = e_new
                best_rows = [rr[:] for rr in rows]
                best_scope = s_new
                if v_new:
                    best_scope = min(best_scope, s_new)
        else:
            rows[ri] = old

        T = max(T_min, T * cooling)

        # occasional full repair to escape deep local minima
        if c_cur > 0 and rng.random() < 0.05:
            cand = _repair(rows, k, rng)
            e_r, v_r, s_r, c_r = energy(cand)
            if e_r < e_cur:
                rows[:] = cand
                e_cur, v_cur, s_cur, c_cur = e_r, v_r, s_r, c_r
                if e_r < e_best:
                    e_best = e_r
                    best_rows = [rr[:] for rr in rows]
                    best_scope = s_r

    return best_rows, best_scope


def construct(n_rows, k):
    deadline = time.time() + _DEADLINE
    rng = random.Random(12345)

    # baseline greedy (always valid)
    used = 0
    greedy_rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        greedy_rows.append(row)
        used |= _diff_mask(row)
    best_rows = [r[:] for r in greedy_rows]
    best_scope = max(r[-1] for r in greedy_rows)

    modes = ["random", "perturbed", "greedy"]
    for mode in modes:
        if time.time() > deadline:
            break
        init = _initialise(n_rows, k, mode, rng)
        rows, scope = _anneal(init, k, rng, deadline)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # final safety: ensure validity of best_rows; if not, fall back to greedy
    v, s, c = _eval(best_rows, k)
    if not v:
        best_rows = [r[:] for r in greedy_rows]
        best_scope = max(r[-1] for r in greedy_rows)

    return best_rows
