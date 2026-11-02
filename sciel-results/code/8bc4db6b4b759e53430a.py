import numpy as np
import time

N = 50
_TIME_LIMIT = 150.0
_start_time = time.time()


def _score(h):
    """Score = ||h||_2^2 / (||h||_1 * sup autoconvolution)."""
    h = np.asarray(h, dtype=float)
    s1 = float(np.sum(h))
    if s1 < 1e-12:
        return 0.0
    s2 = float(np.dot(h, h))
    conv = np.convolve(h, h, mode='full')
    mc = float(np.max(conv))
    if mc < 1e-12:
        return 0.0
    return s2 / (s1 * mc)


def _top_lag(h):
    conv = np.convolve(h, h, mode='full')
    return int(np.argmax(conv))


def _eig_solve(k, S):
    """Maximize Rayleigh quotient for fixed lag k on support S.
    Returns (value, h_full) or (None, None) if singular/invalid."""
    idx = np.asarray(S, dtype=int)
    m = idx.size
    if m == 0:
        return None, None
    I = np.eye(m)
    sym = np.zeros((m, m))
    for a in range(m):
        ia = idx[a]
        for b in range(a + 1, m):
            ib = idx[b]
            if ia + ib == k:
                sym[a, b] = 1.0
                sym[b, a] = 1.0
    A = 0.5 * (I + sym)
    ones = np.ones(m)
    v = np.zeros(m)
    for a in range(m):
        j = k - idx[a]
        if 0 <= j < N:
            v[a] = 1.0
    B = 0.5 * (np.outer(ones, v) + np.outer(v, ones))
    try:
        w, V = np.linalg.eigh(A, B)
    except np.linalg.LinAlgError:
        return None, None
    if not np.isfinite(w).all():
        return None, None
    val = w.max()
    if val <= 0 or not np.isfinite(val):
        return None, None
    x = V[:, int(np.argmax(w))]
    x = np.clip(x, 0.0, None)
    s1 = float(np.sum(x))
    if s1 < 1e-12:
        return None, None
    h = np.zeros(N)
    h[idx] = x / s1
    return val, h


def _support_search(S0, deadline):
    """Coordinate ascent over supports with eigensolves at each step."""
    S = list(S0)
    val, h = _eig_solve(_top_lag(h) if h is not None else 0, S)
    if h is None:
        h = np.zeros(N); h[0] = 1.0
        val = _score(h)
    cur = _score(h)
    curk = _top_lag(h)

    def neighbors():
        cand = set()
        for i in S:
            cand.add(frozenset(S) - {i})
            for j in range(N):
                if j not in S:
                    cand.add((frozenset(S) - {i}) | {j})
        for j in range(N):
            if j not in S:
                cand.add(frozenset(S) | {j})
        return list(cand)

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for ns in neighbors():
            if time.time() > deadline - 2.0:
                break
            ns = list(ns)
            k2 = curk
            v2, h2 = _eig_solve(k2, ns)
            if h2 is None:
                continue
            s2 = _score(h2)
            if s2 > cur + 1e-9:
                cur, h, val = s2, h2, v2
                curk = _top_lag(h)
                S = ns
                improved = True
                break
    return h, cur


def construct():
    deadline = _start_time + _TIME_LIMIT - 8.0

    # --- Phase 1: global enumeration over all lags (exact Rayleigh subproblem) ---
    best_h = np.zeros(N); best_h[0] = 1.0
    best_val = _score(best_h)
    for k in range(2 * N - 1):
        if time.time() > deadline - 30.0:
            break
        v, h = _eig_solve(k, list(range(N)))
        if h is None:
            continue
        s = _score(h)
        if s > best_val:
            best_val, best_h = s, h

    # --- Phase 2: support coordinate ascent from the best global point ---
    S0 = [i for i in range(N) if best_h[i] > 1e-9]
    if not S0:
        S0 = [0]
    h2, v2 = _support_search(S0, deadline)
    if v2 > best_val:
        best_val, best_h = v2, h2

    # --- Phase 3: a few random-support restarts for robustness ---
    rng = np.random.default_rng(1234)
    for _ in range(6):
        if time.time() > deadline - 6.0:
            break
        m = int(rng.integers(3, 15))
        S0 = sorted(rng.choice(N, size=m, replace=False).tolist())
        h3, v3 = _support_search(S0, deadline)
        if v3 > best_val:
            best_val, best_h = v3, h3

    if time.time() > deadline - 2.0 or best_h is None:
        return [float(x) for x in best_h]

    # --- Phase 4: final exact recompute (no extra optimization) ---
    Sfin = [i for i in range(N) if best_h[i] > 1e-9]
    if Sfin:
        kf = _top_lag(best_h)
        vf, hf = _eig_solve(kf, Sfin)
        if hf is not None and _score(hf) > best_val:
            best_val, best_h = _score(hf), hf

    return [float(x) for x in best_h]
