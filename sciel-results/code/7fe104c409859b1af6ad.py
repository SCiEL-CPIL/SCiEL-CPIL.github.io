"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 78.0  # safety margin vs 90s wall-clock kill


def _build_matrix():
    """Incidence matrix M (rows = APs of length>=2, cols = positions 0..255).
    M[k, j] = 1 if position j is in AP k else 0.
    sum over AP k = M[k] @ signs."""
    rows = []
    n = _N
    for a in range(1, n + 1):          # start (1-indexed)
        for d in range(1, n - a + 1):  # step
            # length of AP starting at a with step d within [1, n]
            L = (n - a) // d + 1
            if L >= 2:                 # skip singletons (contribute exactly 1)
                idx = [a - 1 + d * t for t in range(L)]
                row = np.zeros(n, dtype=np.float32)
                row[idx] = 1.0
                rows.append(row)
    M = np.array(rows, dtype=np.float32)
    return M


def _max_abs_sum(s, M):
    """Return (max_abs, worst_indices) over all APs (length>=2).
    Singletons always contribute 1, so overall max = max(1, max|sum|)."""
    sums = M @ s
    am = np.abs(sums)
    k = int(np.argmax(am))
    return float(am[k])


def _best_flip(s, M, worst_idx):
    """Given current signs s and a worst AP index, find the single flip that
    most reduces the objective. Returns (gain, pos) or (None, None).
    gain = current_max - new_max (positive => improvement)."""
    base = _max_abs_sum(s, M)
    n = _N
    best_gain = 0.0
    best_pos = None
    col = M[worst_idx]
    # only flipping a position in the worst AP can change its sum
    cand = np.nonzero(col)[0]
    for p in cand:
        s[p] = -s[p]
        m = _max_abs_sum(s, M)
        s[p] = -s[p]
        g = base - m
        if g > best_gain:
            best_gain = g
            best_pos = int(p)
    if best_pos is None:
        return None, None
    return best_gain, best_pos


def _best_2flip(s, M, worst_idx, limit=4000):
    """Try coordinated 2-flips within the worst AP to escape local minima.
    Returns (gain, p, q) or (None, None, None)."""
    base = _max_abs_sum(s, M)
    col = M[worst_idx]
    cand = np.nonzero(col)[0]
    if len(cand) < 2:
        return None, None, None
    best_gain = 0.0
    best_pq = (None, None)
    L = len(cand)
    for i in range(L):
        p = cand[i]
        for j in range(i + 1, L):
            q = cand[j]
            s[p] = -s[p]
            s[q] = -s[q]
            m = _max_abs_sum(s, M)
            s[p] = -s[p]
            s[q] = -s[q]
            g = base - m
            if g > best_gain:
                best_gain = g
                best_pq = (int(p), int(q))
            if limit is not None and (i * L + j) > limit:
                break
        if limit is not None and i * L > limit:
            break
    if best_pq[0] is None:
        return None, None, None
    return best_gain, best_pq[0], best_pq[1]


def construct():
    t0 = time.time()
    M = _build_matrix()
    rng = np.random.RandomState(171)

    # start from random signs
    s = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=_N).astype(np.float32)
    best_s = s.copy()
    best_val = _max_abs_sum(best_s, M)

    def remaining():
        return _DEADLINE - (time.time() - t0)

    stuck_rounds = 0
    while remaining() > 1.5:
        worst_idx = int(np.argmax(np.abs(M @ s)))
        gain, p = _best_flip(s, M, worst_idx)
        if gain is not None and p is not None:
            s[p] = -s[p]
            stuck_rounds = 0
            v = _max_abs_sum(s, M)
            if v < best_val:
                best_val = v
                best_s = s.copy()
            continue

        # stuck: try a targeted 2-flip in the worst AP
        g2, p2, q2 = _best_2flip(s, M, worst_idx)
        if g2 is not None and p2 is not None and q2 is not None:
            s[p2] = -s[p2]
            s[q2] = -s[q2]
            stuck_rounds = 0
            v = _max_abs_sum(s, M)
            if v < best_val:
                best_val = v
                best_s = s.copy()
            continue

        # fully stuck: random kick to escape basin
        if remaining() < 3.0:
            break
        for _ in range(4):
            kk = int(rng.randint(0, _N))
            s[kk] = -s[kk]
        v = _max_abs_sum(s, M)
        if v < best_val:
            best_val = v
            best_s = s.copy()
        stuck_rounds += 1
        if stuck_rounds > 60:
            # restart from best to avoid drifting
            s = best_s.copy()
            stuck_rounds = 0

    # final safeguard: ensure we return a valid sign vector of length 256
    out = np.where(best_s > 0, 1, -1)
    return [int(x) for x in out]
# EVOLVE-BLOCK-END
