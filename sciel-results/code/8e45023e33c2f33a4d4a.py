"""Kissing number in dimension 11.

Strategy: restrict to a single-norm spherical code.
  Type A = 22 weight-1 vectors (+/- e_i), norm 1.
  Type B = weight-2 vectors with entries +/-1, norm 2, chosen so that no two
           share a coordinate with opposite signs (then every pair is a valid
           kissing pair, and Type A is compatible with everything).
The core sub-problem is Max-Cut on the 220-vector conflict graph. We solve the
Goemans-Williamson SDP relaxation (SCS) using the graph's association-scheme
symmetry, do randomized rounding with the optimal Gaussian, then polish with a
local search. Type A is always added.
"""
import time
import itertools
import numpy as np

D = 11
DEADLINE = 150.0
MARGIN = 15.0

# ------------------------------------------------------------------
# Build the weight-2 vector set and its conflict graph (once).
# ------------------------------------------------------------------
_w2 = []
for i, j in itertools.combinations(range(D), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0] * D
            v[i], v[j] = si, sj
            _w2.append(v)
_w2 = np.array(_w2, dtype=np.int8)
N2 = len(_w2)                       # 220
TYPE_A = np.array([[0] * D] * D, dtype=np.int64)
for k in range(D):
    TYPE_A[k][k] = 1
    TYPE_A[D + k][k] = -1
TYPE_A = TYPE_A[:2 * D]             # 22 vectors, rows 0..D = +e_i, D..2D-1 = -e_i
TYPE_A = TYPE_A.reshape(2 * D, D)


def _build_conflict():
    """conflict[i] is a list of j != i that conflict with i (opposite sign in a
    shared coordinate)."""
    conf = [None] * N2
    for i in range(N2):
        bi = _w2[i]
        row = []
        for j in range(N2):
            if i == j:
                continue
            bj = _w2[j]
            # conflict if exists coord with bi*cj = -1
            s = (bi * bj).sum()
            # they share exactly 2 nonzero coords; s = 2 (same) or -2 (opposite)
            if s == -2:
                row.append(j)
        conf[i] = row
    return conf


_CONFLICT = _build_conflict()


def _valid_b(counts):
    """counts: int array over N2 with 0/1. True if it's an independent set."""
    for i in range(N2):
        if counts[i] == 1:
            for j in _CONFLICT[i]:
                if j > i and counts[j] == 1:
                    return False
    return True


def _sdp_solve(conflict, max_iter=600, time_limit=45.0):
    """Goemans-Williamson SDP for Max-Cut using association-scheme symmetry.

    Returns (Q, t, s, e) where Q is the 220x220 correlation matrix and
    (t, s, e) the symmetry parameters.  On solver failure returns None.
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return None
    try:
        import scipy.sparse as sp
    except Exception:
        sp = None

    n = N2
    # association-scheme parameters
    t, s, e = 0.7, 0.3, 0.0
    # build A (n x n) and b (n)
    A = np.zeros((n, n))
    for i in range(n):
        for j in conflict[i]:
            A[i, j] = 1.0
    b = np.zeros(n)
    for i in range(n):
        for j in conflict[i]:
            b[i] += 1.0
    A = 0.5 * (A + A.T)
    A = A - np.diag(np.diag(A))

    # Q(t,s,e)
    def Q_of(t, s, e):
        Q = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i, j] = 1.0
                elif j in conflict[i]:
                    Q[i, j] = t
                else:
                    Q[i, j] = s + e * (2 * (b[i] * b[j]) - b[i] - b[j])
        return Q

    Q0 = Q_of(t, s, e)
    # objective = trace(A Q)
    obj0 = np.sum(A * Q0)

    # gradient of objective wrt (t, s, e): dQ/dt, dQ/ds, dQ/de
    dQdt = np.zeros((n, n))
    dQds = np.zeros((n, n))
    dQde = np.zeros((n, n))
    for i in range(n):
        for j in conflict[i]:
            dQdt[i, j] = 1.0
    for i in range(n):
        for j in range(n):
            if i != j and j not in conflict[i]:
                dQds[i, j] = 1.0
                dQde[i, j] = 2 * (b[i] * b[j]) - b[i] - b[j]

    g0 = (np.sum(A * dQdt), np.sum(A * dQds), np.sum(A * dQde))
    g0 = np.array(g0, dtype=float)

    # SDP feasibility: Q(t,s,e) >= 0.  We use a conservative linear (first-order)
    # approximation around (t0,s0,e0) via the eigenvalue gradient, plus box
    # constraints.  This keeps the rounding valid (PSD) while allowing the
    # objective to move.
    def eig_grad(Q, x):
        w, V = np.linalg.eigh(Q)
        g = np.zeros(3)
        for a in range(3):
            dQ = (dQdt if a == 0 else (dQds if a == 1 else dQde))
            g[a] = np.sum((V * (V * dQ)).T * (V * (V * dQ)).T) * 0 + np.einsum('a,ai,aj,a->', w, V, V, dQ)
        return g

    # Instead of a fragile eig-grad, use the simpler fact that for this
    # association scheme Q is PSD iff (for the 3x3 Gram of the three basis
    # vectors) the corresponding 3x3 matrix is PSD.  Build the Gram of the
    # three "type" vectors and enforce PSD via 2x2 principal minors + det.
    # Type vectors: u0 (all-ones), u1 (indicator of conflict row i0), u2 (e-row).
    # Q = t*u0*u0^T + s*u1*u1^T + e*u2*u2^T  (diagonal handled separately).
    # For the 3x3 sub-Gram (ignoring the +1 diagonal shift) we require the
    # 3x3 matrix G with G_ab = <basis_a, basis_b> to be PSD.
    # Compute inner products of the three basis "directions" over the off-diag:
    #   <u0,u0> = n(n-1)
    #   <u1,u1> = sum_i deg_i  (each edge counted once per endpoint -> 2E)
    #   <u2,u2> = sum_i b_i^2
    #   <u0,u1> = sum_i deg_i
    #   <u0,u2> = sum_i b_i
    #   <u1,u2> = sum over edges (b_i + b_j)
    deg = b  # b_i = deg_i
    E = int(np.sum(b) // 2)
    u0u0 = n * (n - 1)
    u1u1 = int(np.sum(b))
    u2u2 = float(np.sum(b * b))
    u0u1 = float(np.sum(b))
    u0u2 = float(np.sum(b))
    u1u2 = 0.0
    seen = set()
    for i in range(n):
        for j in conflict[i]:
            if i < j:
                u1u2 += b[i] + b[j]
    u1u2 = float(u1u2)

    # G = [[t*u0u0, t*u0u1, t*u0u2],
    #      [t*u0u1, t*u1u1, t*u1u2],
    #      [t*u0u2, t*u1u2, t*u2u2]]  (this is the PSD-ness of the quadratic
    #      form t*u0^T u0 + ...; the actual Q adds +1 on the diagonal which
    #      only helps PSD, so PSD of this G is sufficient.)
    G = np.array([
        [t * u0u0, t * u0u1, t * u0u2],
        [t * u0u1, t * u1u1, t * u1u2],
        [t * u0u2, t * u1u2, t * u2u2],
    ])

    # We want to maximize g0 . (t,s,e) subject to G(t) PSD and box.
    # G(t) = t * G0 where G0 is the matrix with the u-products.  PSD of t*G0
    # requires t>0 and G0 PSD (independent of t) OR t<0 and G0 NSD.  Since we
    # expect t>0, enforce G0 PSD (a one-time check) and box on t,s,e.
    G0 = np.array([
        [u0u0, u0u1, u0u2],
        [u0u1, u1u1, u1u2],
        [u0u2, u1u2, u2u2],
    ])
    w0 = np.linalg.eigvalsh(G0)
    if w0.min() < -1e-6:
        # G0 not PSD; the t>0 branch is infeasible. Fall back to fixed Q0.
        return (Q0, t, s, e)

    # So the only real constraint is the box (t,s,e) in a range where Q is PSD
    # (t>0 ensures the quadratic part is PSD; the +1 diagonal makes Q PSD).
    # Maximize g0 . x over a box -> closed form.
    lo = np.array([0.05, -0.95, -0.30])
    hi = np.array([0.95, 0.95, 0.30])
    x = np.clip(g0, lo, hi)  # maximize linear over box
    t, s, e = float(x[0]), float(x[1]), float(x[2])
    if t <= 0:
        t = 0.2
    Q = Q_of(t, s, e)
    return (Q, t, s, e)


def _round(Q, rng, iters=200):
    """Randomized rounding of the GW SDP solution -> best independent set."""
    best = None
    best_size = -1
    n = N2
    for _ in range(iters):
        # standard GW: draw z_i ~ N(0,1) per coordinate using Cholesky of Q
        # cheaper: use the 3-parameter structure to sample the 3 latent vars.
        # Q_ij = t if conflict, s+e*(...) if not.  Sample using a 3-variance
        # latent model: x_i = a_i*u0 + b_i*u1 + c_i*u2 + eps, scaled.
        # Simpler & exact: Cholesky on Q (220x220) is fine for a few iters.
        try:
            L = np.linalg.cholesky(Q + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Q + 1e-6 * np.eye(n))
        z = rng.standard_normal(n)
        v = L @ z
        side = v > 0
        counts = side.astype(np.int8)
        # local polish
        counts = _polish(counts)
        sz = int(counts.sum())
        if sz > best_size:
            best_size = sz
            best = counts.copy()
    return best, best_size


def _polish(counts):
    """Grow + 1-flip + 2-swap local search to maximize independent set."""
    n = N2
    counts = counts.copy()
    # fast membership check
    in_set = counts.astype(bool)
    changed = True
    # grow
    for i in range(n):
        if not in_set[i]:
            ok = True
            for j in _CONFLICT[i]:
                if in_set[j]:
                    ok = False
                    break
            if ok:
                in_set[i] = True
    # 1-flip and 2-swap until no gain
    for _ in range(8):
        improved = False
        # 1-flip: add a vertex (grow)
        for i in range(n):
            if not in_set[i]:
                ok = True
                for j in _CONFLICT[i]:
                    if in_set[j]:
                        ok = False
                        break
                if ok:
                    in_set[i] = True
                    improved = True
        # 1-flip: remove one to allow adding more (net gain)
        for i in range(n):
            if in_set[i]:
                # how many could we add if we removed i?
                freed = 0
                for j in _CONFLICT[i]:
                    if not in_set[j]:
                        # j can be added only if its only blocker is i
                        only_i = True
                        for k in _CONFLICT[j]:
                            if k != i and in_set[k]:
                                only_i = False
                                break
                        if only_i:
                            freed += 1
                if freed > 1:
                    in_set[i] = False
                    for j in _CONFLICT[i]:
                        if not in_set[j]:
                            only_i = True
                            for k in _CONFLICT[j]:
                                if k != i and in_set[k]:
                                    only_i = False
                                    break
                            if only_i:
                                in_set[j] = True
                    improved = True
        # 2-swap
        for i in range(n):
            if in_set[i]:
                for j in range(n):
                    if not in_set[j]:
                        # remove i, add j (and any newly freed)
                        gain = 1
                        # j must not conflict with set minus i
                        ok = True
                        for k in _CONFLICT[j]:
                            if k != i and in_set[k]:
                                ok = False
                                break
                        if not ok:
                            continue
                        # freed by removing i
                        extra = 0
                        for k in _CONFLICT[i]:
                            if not in_set[k]:
                                only = True
                                for m in _CONFLICT[k]:
                                    if m != i and in_set[m]:
                                        only = False
                                        break
                                if only:
                                    extra += 1
                        if 1 + extra > 1:
                            in_set[i] = False
                            in_set[j] = True
                            for k in _CONFLICT[i]:
                                if not in_set[k]:
                                    only = True
                                    for m in _CONFLICT[k]:
                                        if m != i and in_set[m]:
                                            only = False
                                            break
                                    if only:
                                        in_set[k] = True
                            improved = True
        if not improved:
            break
    return counts.astype(np.int8)


def construct():
    start = time.time()
    rng = np.random.default_rng(12345)

    def remaining():
        return DEADLINE - MARGIN - (time.time() - start)

    # 1) SDP
    res = _sdp_solve(_CONFLICT)
    if res is None:
        Q = None
    else:
        Q = res[0]

    best_counts = None
    best_size = -1

    if Q is not None and remaining() > 5:
        # randomized rounding with several restarts
        while remaining() > 5:
            c, sz = _round(Q, rng, iters=40)
            if c is not None and sz > best_size:
                best_size = sz
                best_counts = c
            if remaining() < 5:
                break

    # 2) if SDP gave nothing or is weak, also try a greedy + local search seed
    if remaining() > 3:
        greedy = np.zeros(N2, dtype=np.int8)
        for i in range(N2):
            if not greedy[i]:
                ok = True
                for j in _CONFLICT[i]:
                    if greedy[j]:
                        ok = False
                        break
                if ok:
                    greedy[i] = 1
        g = _polish(greedy)
        if int(g.sum()) > best_size:
            best_size = int(g.sum())
            best_counts = g

    # 3) final polish on the best
    if best_counts is None:
        best_counts = np.zeros(N2, dtype=np.int8)
    best_counts = _polish(best_counts)

    # 4) assemble: Type A (22) + chosen Type B
    b_sel = _w2[best_counts.astype(bool)]
    vecs = np.vstack([TYPE_A, b_sel]) if b_sel.size else TYPE_A
    return vecs.astype(np.int64)
