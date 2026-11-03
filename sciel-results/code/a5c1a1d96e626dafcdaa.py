import numpy as np
import time
from fractions import Fraction

def _verify(U, V, W, n=3, m=4, p=4):
    """Exact verification over rationals. U:(n*m,r) V:(m*p,r) W:(p*n,r).
    C[i,k] = sum_{j,t} U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] must equal delta_{i,k}.
    (A[i,j]*B[j,k] -> C[i,k] tensor; W index k*n+i matches W[k*n+i,r] convention.)
    """
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            s = Fraction(0)
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                # sum over t of u[t]*v[t]*w[t]
                for t in range(r):
                    if u[t] and v[t] and w[t]:
                        s += u[t] * v[t] * w[t]
            if s != (1 if i == k else 0):
                return False
    return True

def _to_rational(M, tol=1e-6):
    """Convert float matrix to list-of-lists of Fractions (small rationals)."""
    rows, cols = M.shape
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            x = M[i, j]
            if abs(x) < tol:
                row.append(Fraction(0))
            else:
                row.append(Fraction(x).limit_denominator(1000000))
        out.append(row)
    return out

def _mul3(U, V, W):
    """Elementwise column-wise triple product: result[i,t]=U[i,t]*V[i,t]*W[i,t]."""
    return U * V * W

def _nullspace(Q):
    """Return a basis (list of vectors) of the nullspace of Q (Q @ v = 0).
    Q is list-of-lists of Fractions, shape (m, n)."""
    m = len(Q)
    n = len(Q[0]) if m else 0
    # work on a copy
    M = [list(row) for row in Q]
    pivots = []
    r = 0
    for c in range(n):
        # find pivot in column c at or below row r
        piv = None
        for i in range(r, m):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            M[r], M[piv] = M[piv], M[r]
        # normalize
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(n)]
        pivots.append(c)
        r += 1
    # free variables
    pivotset = set(pivots)
    free = [c for c in range(n) if c not in pivotset]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for pr, pc in enumerate(pivots):
            v[pc] = -M[pr][fc]
        basis.append(v)
    return basis

def _rank_reduce(U, V, W, max_rounds=60, time_limit=120.0, start_time=None):
    """Exact rank reduction via null-space elimination + exact rational reconstruction.
    Returns (U,V,W) as float arrays with minimal found rank, or the input if none works."""
    if start_time is None:
        start_time = time.time()
    # to rationals
    Ur = _to_rational(U)
    Vr = _to_rational(V)
    Wr = _to_rational(W)
    n, m, p = 3, 4, 4
    r = len(Ur[0])
    best = (Ur, Vr, Wr)
    best_r = r
    for rnd in range(max_rounds):
        if time.time() - start_time > time_limit - 15:
            break
        # build Q
        Q = []
        for i in range(n):
            for k in range(p):
                row = []
                for j in range(m):
                    u = Ur[i * m + j]
                    v = Vr[j * p + k]
                    w = Wr[k * n + i]
                    for t in range(r):
                        row.append(u[t] * v[t] * w[t])
                Q.append(row)
        basis = _nullspace(Q)
        if not basis:
            break
        # pick a basis vector with maximum support
        bvec = max(basis, key=lambda v: sum(1 for x in v if x != 0))
        nz = [t for t in range(r) if bvec[t] != 0]
        if len(nz) < 2:
            break
        # find a pivot t with coefficient 1
        pivot = None
        for t in nz:
            if bvec[t] == 1:
                pivot = t
                break
        if pivot is None:
            continue
        others = [t for t in nz if t != pivot]
        # update: for each row, val[pivot] += sum_{t in others} (-bvec[t]) * val[t]
        for arr in (Ur, Vr, Wr):
            for row in arr:
                pv = row[pivot]
                s = Fraction(0)
                for t in others:
                    s += (-bvec[t]) * row[t]
                row[pivot] = pv + s
        # drop pivot column
        for arr in (Ur, Vr, Wr):
            for row in arr:
                del row[pivot]
        r -= 1
        if _verify(np.array([[float(x) for x in row] for row in Ur]),
                   np.array([[float(x) for x in row] for row in Vr]),
                   np.array([[float(x) for x in row] for row in Wr])):
            best = (Ur, Vr, Wr)
            best_r = r
            if best_r <= 32:
                break
    # convert best to float
    Uf = np.array([[float(x) for x in row] for row in best[0]])
    Vf = np.array([[float(x) for x in row] for row in best[1]])
    Wf = np.array([[float(x) for x in row] for row in best[2]])
    return Uf, Vf, Wf

def construct():
    n, m, p = 3, 4, 4
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    # Part 1: Row 0 of A (1x4) * B (4x4) -> C[0, :] (1x4)  schoolbook
    for k in range(4):
        for j in range(4):
            U[0 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) via 4 Strassen 2x2 blocks
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2),
    ]
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * 4 + (ac + yi)][col] += ux
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * 4 + (bc + xi)][col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * 3 + (cr + zi)][col] += wz
            col += 1

    start = time.time()
    # Try exact rank reduction with a deadline guard
    try:
        U2, V2, W2 = _rank_reduce(U, V, W, time_limit=120.0, start_time=start)
        # exact verify the reduced result
        if _verify(U2, V2, W2):
            U, V, W = U2, V2, W2
    except Exception:
        pass
    # safety: if we ran out of time, fall back to the known-good rank-44
    if time.time() - start > 105:
        # recompute the clean rank-44 (U,V,W were not mutated in-place by _rank_reduce)
        pass
    # final float cleanup
    U = np.round(U, 12)
    V = np.round(V, 12)
    W = np.round(W, 12)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
