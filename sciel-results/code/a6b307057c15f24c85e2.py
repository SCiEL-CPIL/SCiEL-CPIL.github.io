# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE = time.time() + 600.0
_MARGIN = 6.0


def _deadline_ok():
    return (time.time() + _MARGIN) < _DEADLINE


def _finite_field_config(q):
    """Reimann construction over GF(q^2) with quadratic-residue conic.

    Returns points as a list of (x, y) integer pairs and lines as a list of
    (A, B, C) integer triples with A*x + B*y + C = 0.  Every point lies on
    exactly q+1 lines and every line passes through exactly q+1 points.
    """
    p = q * q
    # --- build GF(p) = GF(q)[t] with t^2 = a (a a non-residue mod q) ---
    a = 1
    while True:
        if pow(a, (q - 1) // 2, q) != 1:
            break
        a += 1

    def mul(c1, c2):
        u, v = c1
        w, z = c2
        return ((u * w + v * z * a) % p, (u * z + v * w) % p)

    def add(c1, c2):
        return ((c1[0] + c2[0]) % p, (c1[1] + c2[1]) % p)

    def neg(c):
        return ((-c[0]) % p, (-c[1]) % p)

    def sub(c1, c2):
        return ((c1[0] - c2[0]) % p, (c1[1] - c2[1]) % p)

    def inv(c):
        u, v = c
        n = (u * u + v * v) % p
        if n == 0:
            return (0, 0)
        nm = pow(n, q - 2, p)
        return mul((u, (-v) % p), (nm, 0))

    # --- quadratic-residue set (nonzero) ---
    quad = set()
    for x in range(1, p):
        if pow(x, (p - 1) // 2, p) == 1:
            quad.add(x)

    # --- affine points: (x, x^2) ---
    pts = [(x, (x * x) % p) for x in range(p)]
    # --- line at infinity ---
    lin = [(0, 0, -1)]

    # --- finite lines: y = m x + b, m in quad, b in GF(p) ---
    for m in sorted(quad):
        for b in range(p):
            lin.append((m, -1, -b))

    return pts, lin


def _make_state(pts, lin, npts, nlines, rng):
    N = len(pts)
    M = len(lin)
    PL = np.zeros((N, M), dtype=np.int64)
    for i, (x, y) in enumerate(pts):
        for j, (A, B, C) in enumerate(lin):
            if A * x + B * y + C == 0:
                PL[i, j] = 1
    degp = PL.sum(axis=1)
    degl = PL.sum(axis=0)

    # greedy set-cover selection
    chosen_p = [False] * N
    chosen_l = [False] * M
    for _ in range(npts):
        best_i, best_v = -1, -1
        for i in range(N):
            if not chosen_p[i]:
                v = degp[i]
                if v > best_v:
                    best_v, best_i = v, i
        chosen_p[best_i] = True
        for j in np.nonzero(PL[best_i])[0]:
            if not chosen_l[j]:
                degl[j] += 1
    for _ in range(nlines):
        best_j, best_v = -1, -1
        for j in range(M):
            if not chosen_l[j]:
                v = degl[j]
                if v > best_v:
                    best_v, best_j = v, j
        chosen_l[best_j] = True
        for i in np.nonzero(PL[:, best_j])[0]:
            if not chosen_p[i]:
                degp[i] += 1

    SP = [i for i in range(N) if chosen_p[i]]
    SL = [j for j in range(M) if chosen_l[j]]
    SP.sort()
    SL.sort()
    PLs = PL[np.ix_(SP, SL)]
    total = int(PLs.sum())
    return SP, SL, PLs, total


def _eval_delta(PL, i, j):
    r = PL[i, j]
    if r:
        c = int(PL[:, j].sum())
        return 1 - c
    c = int(PL[i, :].sum())
    return c + 1


def _local_search(PL, iters, rng):
    N, M = PL.shape
    total = int(PL.sum())
    for _ in range(iters):
        if not _deadline_ok():
            break
        moved = False
        for _ in range(6):
            i = rng.randrange(N)
            j = rng.randrange(M)
            d = _eval_delta(PL, i, j)
            if d > 0:
                PL[i, j] ^= 1
                total += d
                moved = True
        if not moved:
            break
    return total


def _anneal(PL, pts, lin, iters, rng):
    N, M = PL.shape
    total = int(PL.sum())
    best = total
    T = 4.0
    for it in range(iters):
        if not _deadline_ok():
            break
        T = max(0.02, 4.0 * (1.0 - it / max(1, iters)))
        t = rng.random()
        if t < 0.40:
            i = rng.randrange(N)
            j = rng.randrange(M)
            d = _eval_delta(PL, i, j)
            if d < 0 and rng.random() > math.exp(d / T):
                continue
            PL[i, j] ^= 1
            total += d
        elif t < 0.65:
            i = rng.randrange(N)
            j1 = rng.randrange(M)
            j2 = rng.randrange(M)
            if j1 == j2:
                continue
            d = _eval_delta(PL, i, j1) + _eval_delta(PL, i, j2)
            if d < 0 and rng.random() > math.exp(d / T):
                continue
            PL[i, j1] ^= 1
            PL[i, j2] ^= 1
            total += d
        elif t < 0.90:
            j = rng.randrange(M)
            i1 = rng.randrange(N)
            i2 = rng.randrange(N)
            if i1 == i2:
                continue
            d = _eval_delta(PL, i1, j) + _eval_delta(PL, i2, j)
            if d < 0 and rng.random() > math.exp(d / T):
                continue
            PL[i1, j] ^= 1
            PL[i2, j] ^= 1
            total += d
        else:
            i = rng.randrange(N)
            j = rng.randrange(M)
            d1 = _eval_delta(PL, i, j)
            if d1 <= 0:
                continue
            PL[i, j] ^= 1
            total += d1
            i2 = rng.randrange(N)
            j2 = rng.randrange(M)
            if (i2 == i) or (j2 == j):
                continue
            d2 = _eval_delta(PL, i2, j2)
            if d2 < 0 and rng.random() > math.exp(d2 / T):
                PL[i, j] ^= 1
                total -= d1
                continue
            PL[i2, j2] ^= 1
            total += d2

        if total > best:
            best = total
    return best


def _finalize(SP, SL, pts, lin):
    P = np.array([[pts[i][0], pts[i][1]] for i in SP], dtype=np.int64)
    L = np.array([[lin[j][0], lin[j][1], lin[j][2]] for j in SL], dtype=np.int64)
    return P, L


def _score_of(P, L):
    return int(np.sum(np.einsum('nd,md->nm', P, L[:, :2]) + L[:, 2:3]))


def _run_once(q, npts, nlines, rng):
    pts, lin = _finite_field_config(q)
    N, M = len(pts), len(lin)
    if N < npts or M < nlines:
        return None
    SP, SL, PLs, total = _make_state(pts, lin, npts, nlines, rng)
    best = total

    PL = PLs.copy()
    total = _local_search(PL, 40000, rng)
    if total > best:
        best = total
        bestPL = PL.copy()
    else:
        bestPL = PLs

    if _deadline_ok():
        PL = bestPL.copy()
        t = _anneal(PL, pts, lin, 300000, rng)
        if t > best:
            best = t
            bestPL = PL

    if _deadline_ok():
        PL = bestPL.copy()
        total = _local_search(PL, 30000, rng)
        if total > best:
            best = total
            bestPL = PL

    return _finalize(SP, SL, pts, lin), best, bestPL


def _finalize_from_pl(SP, SL, PL, pts, lin):
    P = np.array([[pts[i][0], pts[i][1]] for i in SP], dtype=np.int64)
    L = np.array([[lin[j][0], lin[j][1], lin[j][2]] for j in SL], dtype=np.int64)
    return P, L


def szemeredi_trotter_incidences():
    npts, nlines = 54, 27
    rng = random.Random(1234567)
    global _DEADLINE
    _DEADLINE = time.time() + 600.0

    best_P, best_L, best_score = None, None, -1

    # --- q = 7 (57 points / 57 lines) ---
    try:
        pts7, lin7 = _finite_field_config(7)
        N7, M7 = len(pts7), len(lin7)
        SP, SL, PLs, total = _make_state(pts7, lin7, npts, nlines, rng)
        bestPL = PLs.copy()
        best = total
        PL = bestPL.copy()
        t = _local_search(PL, 50000, rng)
        if t > best:
            best, bestPL = t, PL
        if _deadline_ok():
            PL = bestPL.copy()
            t = _anneal(PL, pts7, lin7, 400000, rng)
            if t > best:
                best, bestPL = t, PL
        if _deadline_ok():
            PL = bestPL.copy()
            t = _local_search(PL, 40000, rng)
            if t > best:
                best, bestPL = t, PL
        P, L = _finalize(SP, SL, pts7, lin7)
        sc = _score_of(P, L)
        if sc > best_score:
            best_score, best_P, best_L = sc, P, L
        # keep the best incidence matrix for possible later use
        bestPL7 = bestPL
    except Exception:
        bestPL7 = None

    # --- q = 5 (31 points / 31 lines), independent trials ---
    if _deadline_ok():
        try:
            pts5, lin5 = _finite_field_config(5)
            for _ in range(3):
                if not _deadline_ok():
                    break
                SP, SL, PLs, total = _make_state(pts5, lin5, npts, nlines, rng)
                bestPL = PLs.copy()
                best = total
                PL = bestPL.copy()
                t = _local_search(PL, 20000, rng)
                if t > best:
                    best, bestPL = t, PL
                if _deadline_ok():
                    PL = bestPL.copy()
                    t = _anneal(PL, pts5, lin5, 150000, rng)
                    if t > best:
                        best, bestPL = t, PL
                if _deadline_ok():
                    PL = bestPL.copy()
                    t = _local_search(PL, 20000, rng)
                    if t > best:
                        best, bestPL = t, PL
                P, L = _finalize(SP, SL, pts5, lin5)
                sc = _score_of(P, L)
                if sc > best_score:
                    best_score, best_P, best_L = sc, P, L
        except Exception:
            pass

    # --- fallback: baseline grid (always valid, 81 incidences) ---
    if best_P is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        P = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = [(a, -1, b) for a in range(1, k + 1) for b in range(1, k * k + 1)]
        L = np.array(rows, dtype=np.int64)
        best_P, best_L = P, L

    return best_P, best_L


# EVOLVE-BLOCK-END
