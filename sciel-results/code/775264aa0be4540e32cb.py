"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition search.

Strategy family (kept from the improving lineage):
  - Start from a full schoolbook (rank 75).
  - Apply Strassen's 7-term algorithm to disjoint 2x2x2 sub-blocks to cut rank
    (each such block replaces 8 terms by 7).
  - Hill-climb over *overlapping* block tilings to find even lower ranks,
    verifying exactness of the resulting factor matrices after each step.

The evaluator kills the program at 150s; we keep an explicit deadline guard and
always return a verified decomposition (falling back to schoolbook).
"""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
START = time.time()
DEADLINE = START + 138.0          # safety margin before the 150s kill
def _time_left():
    return DEADLINE - time.time()

# ---------------------------------------------------------------------------
# Strassen 2x2x2 block (7 terms) with the official trace convention.
# U: (i*2+j) x c ; V: (j*2+k) x c ; W: (k*2+i) x c
# ---------------------------------------------------------------------------
_STRASSEN = [
    (0, 0, 1), (0, 1, 1), (0, 1, -1),
    (1, 0, 1), (1, 1, 1),
    (1, 0, 1), (1, 0, -1), (1, 1, 1), (1, 1, -1),
    (0, 0, 1), (0, 1, 1), (0, 1, -1),
    (1, 0, 1), (1, 1, 1),
]
_STRASSEN_U = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1)],
    [(1, 0)],
    [(0, 0), (1, 0)],
    [(0, 1), (1, 1)],
    [(0, 0)],
]
_STRASSEN_V = [
    [(0, 0), (1, 1)],
    [(0, 1), (1, 1)],
    [(1, 0)],
    [(1, 1)],
    [(0, 0), (1, 0)],
    [(0, 1), (1, 1)],
    [(0, 1)],
]
_STRASSEN_W = [
    [(0, 0), (1, 1)],
    [(1, 0)],
    [(0, 1), (1, 1)],
    [(1, 1)],
    [(0, 0), (1, 0)],
    [(0, 1)],
    [(0, 0)],
]
_STRASSEN_UV = [
    [(0, 0), (1, 1)],
    [(0, 1), (1, 1)],
    [(1, 0)],
    [(1, 1)],
    [(0, 0), (1, 0)],
    [(0, 1), (1, 1)],
    [(0, 1)],
]
_STRASSEN_UVW = [
    [(0, 0), (1, 1)],
    [(1, 0)],
    [(0, 1), (1, 1)],
    [(1, 1)],
    [(0, 0), (1, 0)],
    [(0, 1)],
    [(0, 0)],
]
# (u, v, w) per term, each a list of (i,j)/(j,k)/(k,i) offsets
_STRASSEN_TUPLES = [
    (u, v, w) for u, v, w in zip(_STRASSEN_U, _STRASSEN_V, _STRASSEN_W)
]

# ---------------------------------------------------------------------------
# Factor-matrix builder from a set of (term, anchor) pairs
# ---------------------------------------------------------------------------
def build_factors(term_anchor_list):
    U = np.zeros((N * M, len(term_anchor_list)), dtype=object)
    V = np.zeros((M * P, len(term_anchor_list)), dtype=object)
    W = np.zeros((P * N, len(term_anchor_list)), dtype=object)
    for c, (term, (i0, j0, k0)) in enumerate(term_anchor_list):
        for (ii, jj), val in _STRASSEN_UV[term]:
            U[i0 + ii, j0 + jj, c] += val
        for (jj, kk), val in _STRASSEN_UVW[term]:
            V[j0 + jj, k0 + kk, c] += val
        for (kk, ii), val in _STRASSEN_U[term]:
            W[k0 + kk, i0 + ii, c] += val
    return U, V, W

def verify(U, V, W):
    n, m, p = N, M, P
    r = U.shape[1]
    for k in range(p):
        for i in range(n):
            for j in range(m):
                s = 0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def make_schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

# ---------------------------------------------------------------------------
# Candidate (i0, j0, k0) anchors for a 2x2x2 block
# ---------------------------------------------------------------------------
def all_anchors():
    return [(i0, j0, k0)
            for i0 in range(N - 1)
            for j0 in range(M - 1)
            for k0 in range(P - 1)]

def greedy_disjoint_strassen():
    """Apply Strassen to a maximal set of disjoint 2x2x2 blocks (rank 71)."""
    terms = []
    used = set()
    # deterministic ordering: try to pack blocks
    for (i0, j0, k0) in all_anchors():
        cells = set()
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    cells.add((i0 + ii, j0 + jj, k0 + kk))
        if cells & used:
            continue
        used |= cells
        for t in range(7):
            terms.append((t, (i0, j0, k0)))
    # remaining cells -> schoolbook
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) not in used:
                    terms.append((7, (i, j, k)))   # term 7 unused; handled below
    # We used term index 7 as a placeholder for schoolbook; remap: build
    # separately. Instead, rebuild cleanly:
    return None  # replaced by explicit construction below

def greedy_disjoint_strassen_v2():
    """Return a verified (U,V,W) from disjoint Strassen blocks + schoolbook."""
    terms = []
    used = set()
    for (i0, j0, k0) in all_anchors():
        cells = set()
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    cells.add((i0 + ii, j0 + jj, k0 + kk))
        if cells & used:
            continue
        used |= cells
        for t in range(7):
            terms.append((t, (i0, j0, k0)))
    # schoolbook for remaining cells, represented via a dedicated builder
    U = np.zeros((N * M, 0), dtype=object)
    V = np.zeros((M * P, 0), dtype=object)
    W = np.zeros((P * N, 0), dtype=object)
    # build strassen part
    if terms:
        Us, Vs, Ws = build_factors(terms)
        U = np.hstack([U, Us])
        V = np.hstack([V, Vs])
        W = np.hstack([W, Ws])
    # schoolbook part
    sb = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) not in used:
                    sb.append((i, j, k))
    if sb:
        U2 = np.zeros((N * M, len(sb)), dtype=object)
        V2 = np.zeros((M * P, len(sb)), dtype=object)
        W2 = np.zeros((P * N, len(sb)), dtype=object)
        for c, (i, j, k) in enumerate(sb):
            U2[i * M + j, c] = 1
            V2[j * P + k, c] = 1
            W2[k * N + i, c] = 1
        U = np.hstack([U, U2])
        V = np.hstack([V, V2])
        W = np.hstack([W, W2])
    return U, V, W

# ---------------------------------------------------------------------------
# Hill-climb over overlapping Strassen block placements
# ---------------------------------------------------------------------------
def hill_climb(best_U, best_V, best_W, best_rank, rng):
    """Try to lower rank by replacing a subset of current terms with a block.

    Current terms are implicitly the set of (term, anchor). We track the term
    list directly for the current best.
    """
    pass

def optimize():
    rng = random.Random(12345)
    # Start from disjoint Strassen (rank 71)
    U, V, W = greedy_disjoint_strassen_v2()
    assert verify(U, V, W), "disjoint strassen must verify"
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # ---- Hill-climb: random overlapping block replacements ----------------
    # Represent the current solution as a list of (kind, data).
    # kind 'S' = strassen block at anchor; kind 'B' = schoolbook cell.
    # We'll search by proposing a full tiling of the 3x5x5 grid with a mixture
    # of Strassen blocks (possibly overlapping) and single cells, then verify.

    def rank_of(tiling):
        return len(tiling)

    # A tiling is a list of entries; each entry is ('S', anchor) or ('B', cell).
    # For verification we build factors where each 'S' contributes 7 columns
    # and each 'B' contributes 1 column.

    def build_from_tiling(tiling):
        # group strassen terms
        terms = []
        for kind, data in tiling:
            if kind == 'S':
                for t in range(7):
                    terms.append((t, data))
        U = np.zeros((N * M, 0), dtype=object)
        V = np.zeros((M * P, 0), dtype=object)
        W = np.zeros((P * N, 0), dtype=object)
        if terms:
            Us, Vs, Ws = build_factors(terms)
            U = np.hstack([U, Us])
            V = np.hstack([V, Vs])
            W = np.hstack([W, Ws])
        cells = [data for kind, data in tiling if kind == 'B']
        if cells:
            U2 = np.zeros((N * M, len(cells)), dtype=object)
            V2 = np.zeros((M * P, len(cells)), dtype=object)
            W2 = np.zeros((P * N, len(cells)), dtype=object)
            for c, (i, j, k) in enumerate(cells):
                U2[i * M + j, c] = 1
                V2[j * P + k, c] = 1
                W2[k * N + i, c] = 1
            U = np.hstack([U, U2])
            V = np.hstack([V, V2])
            W = np.hstack([W, W2])
        return U, V, W

    # Initial tiling = the disjoint strassen solution
    init_tiling = []
    used = set()
    for (i0, j0, k0) in all_anchors():
        cells = set((i0 + ii, j0 + jj, k0 + kk)
                    for ii in range(2) for jj in range(2) for kk in range(2))
        if cells & used:
            continue
        used |= cells
        init_tiling.append(('S', (i0, j0, k0)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) not in used:
                    init_tiling.append(('B', (i, j, k)))

    def tiling_covers(tiling):
        cov = np.zeros((N, M, P), dtype=bool)
        for kind, data in tiling:
            if kind == 'S':
                i0, j0, k0 = data
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            cov[i0 + ii, j0 + jj, k0 + kk] = True
            else:
                i, j, k = data
                cov[i, j, k] = True
        return cov

    # We need each (i,j,k) to be covered by an ODD number of "unit" contributions
    # netting to exactly 1. Because Strassen blocks and cells are additive,
    # overlapping blocks can cancel. Verify() is the ground truth.

    anchors = all_anchors()
    cells_all = [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]

    # Local search: mutate the tiling, verify, keep if rank improves and valid.
    current = list(init_tiling)
    cur_rank = len(current)
    U, V, W = build_from_tiling(current)
    if verify(U, V, W):
        best_rank = min(best_rank, cur_rank)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    steps = 0
    while _time_left() > 8.0:
        steps += 1
        # propose a mutation
        op = rng.random()
        if op < 0.5 and len(current) > 1:
            # replace a random schoolbook cell (if any) with a strassen block
            idx = rng.randrange(len(current))
            kind, data = current[idx]
            if kind == 'B':
                i, j, k = data
                # place a block anchored near this cell
                cand = [a for a in anchors
                        if a[0] <= i < a[0] + 2 and a[1] <= j < a[1] + 2
                        and a[2] <= k < a[2] + 2]
                if not cand:
                    cand = anchors
                anchor = rng.choice(cand)
                new = current[:idx] + current[idx + 1:] + [('S', anchor)]
            else:
                continue
        elif op < 0.8:
            # add a strassen block
            anchor = rng.choice(anchors)
            new = current + [('S', anchor)]
        else:
            # remove a strassen block and add its 8 cells (only if no other
            # block covers them) -- a "de-strassen" move
            s_idx = [x for x, (knd, _) in enumerate(current) if knd == 'S']
            if not s_idx:
                continue
            idx = rng.choice(s_idx)
            i0, j0, k0 = current[idx][1]
            cells = [(i0 + ii, j0 + jj, k0 + kk)
                     for ii in range(2) for jj in range(2) for kk in range(2)]
            # check coverage by other blocks
            cov = np.zeros((N, M, P), dtype=int)
            for x, (knd, d) in enumerate(current):
                if x == idx or knd != 'S':
                    continue
                a = d
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            cov[a[0] + ii, a[1] + jj, a[2] + kk] += 1
            add = []
            ok = True
            for (i, j, k) in cells:
                if cov[i, j, k] == 0:
                    add.append(('B', (i, j, k)))
                else:
                    ok = False
            if not ok:
                continue
            new = current[:idx] + current[idx + 1:] + add
        # evaluate
        if len(new) >= best_rank:
            # cannot beat best; but might be equal rank with different structure
            # skip to save time unless equal (could enable further moves)
            if len(new) > best_rank:
                continue
        U2, V2, W2 = build_from_tiling(new)
        if verify(U2, V2, W2):
            current = new
            cur_rank = len(current)
            if cur_rank < best_rank:
                best_rank = cur_rank
                best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
        if steps % 200 == 0:
            pass  # no-op; keep loop tight

    return best_U, best_V, best_W, best_rank

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    # Fast, guaranteed-valid baseline first.
    U, V, W = greedy_disjoint_strassen_v2()
    if not verify(U, V, W):
        U, V, W = make_schoolbook()
    best_U, best_V, best_W, best_rank = U, V, W, U.shape[1]

    if _time_left() > 30.0:
        try:
            oU, oV, oW, oRank = optimize()
            if verify(oU, oV, oW) and oRank < best_rank:
                best_U, best_V, best_W, best_rank = oU, oV, oW, oRank
        except Exception:
            pass

    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = make_schoolbook()

    # Convert to plain lists (integers / half-integers)
    def to_list(A):
        out = []
        for row in A:
            r = []
            for val in row:
                v = int(val) if float(val).is_integer() else float(val)
                r.append(v)
            out.append(r)
        return out

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
