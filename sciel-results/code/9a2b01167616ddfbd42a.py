import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
DEADLINE = 140.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition. U:(nm,rank) V:(mp,rank) W:(pn,rank).
    W index convention: W[k*n + i, c]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    rec = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.allclose(rec, np.ones((n, m, p)), atol=tol)


def _verify_fast(U, V, W, rank, tol=1e-9):
    """Verify assuming arrays already reshaped: U:(n,m,rank) V:(m,p,rank) W:(p,n,rank)."""
    rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(rec, np.ones((N, M, P)), atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 tensors for 2x2 x 2x2 -> 2x2.
    U[i,j,c], V[j,k,c], W[k,i,c] with i,j,k in {0,1}.
    Returns U,V,W each shape (4,7) flattened row-major (i,j),(j,k),(k,i)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    def setU(i, j, c, v): U[i * 2 + j, c] += v
    def setV(j, k, c, v): V[j * 2 + k, c] += v
    def setW(k, i, c, v): W[k * 2 + i, c] += v

    # m0 = (a00+a11)(b00+b11)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, 1)
    # m1 = (a10+a11)b00
    setU(1, 0, 1, 1); setU(1, 1, 1, 1)
    setV(0, 0, 1, 1)
    setW(0, 1, 1, 1); setW(1, 1, 1, -1)
    # m2 = a00(b01-b11)
    setU(0, 0, 2, 1)
    setV(0, 1, 2, 1); setV(1, 1, 2, -1)
    setW(1, 0, 2, 1); setW(1, 1, 2, 1)
    # m3 = a11(b10-b00)
    setU(1, 1, 3, 1)
    setV(1, 0, 3, 1); setV(0, 0, 3, -1)
    setW(0, 1, 3, 1); setW(1, 1, 3, 1)
    # m4 = (a00+a01)b11
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(1, 1, 4, 1)
    setW(1, 0, 4, -1); setW(1, 1, 4, 1)
    # m5 = (a10-a00)(b00+b01)
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1); setV(0, 1, 5, 1)
    setW(0, 1, 5, 1)
    # m6 = (a01-a11)(b10+b11)
    setU(0, 1, 6, 1); setU(1, 1, 6, -1)
    setV(1, 0, 6, 1); setV(1, 1, 6, 1)
    setW(1, 0, 6, 1)

    rec = np.einsum('ijc,jkc,kic->ijk', U.reshape(2, 2, 7),
                    V.reshape(2, 2, 7), W.reshape(2, 2, 7))
    assert np.allclose(rec, np.ones((2, 2, 2)), atol=1e-12), "Strassen 2x2 broken"
    return U, V, W


def _build_4x4x4_strassen():
    """Rank-28 decomposition for 4x4 x 4x4 -> 4x4 via block Strassen.
    4x4 split into four 2x2 blocks. Block-level Strassen (7 terms),
    each term multiplies two 2x2 blocks (schoolbook rank 4). Total 7*4=28.
    Returns U:(16,28) V:(16,28) W:(16,28)."""
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))

    Uforms = [
        {('A', 0, 0): 1, ('A', 1, 1): 1},
        {('A', 1, 0): 1, ('A', 1, 1): 1},
        {('A', 0, 0): 1},
        {('A', 1, 1): 1},
        {('A', 0, 0): 1, ('A', 0, 1): 1},
        {('A', 1, 0): 1, ('A', 0, 0): -1},
        {('A', 0, 1): 1, ('A', 1, 1): -1},
    ]
    Vforms = [
        {('B', 0, 0): 1, ('B', 1, 1): 1},
        {('B', 0, 0): 1},
        {('B', 0, 1): 1, ('B', 1, 1): -1},
        {('B', 1, 0): 1, ('B', 0, 0): -1},
        {('B', 1, 1): 1},
        {('B', 0, 0): 1, ('B', 0, 1): 1},
        {('B', 1, 0): 1, ('B', 1, 1): 1},
    ]
    Wforms = [
        {('C', 0, 0): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 1, 1): -1},
        {('C', 0, 1): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 0, 0): 1},
        {('C', 0, 1): -1, ('C', 1, 1): 1},
        {('C', 1, 1): 1},
        {('C', 0, 0): 1},
    ]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                for (tag, br, bc), coeff in Uforms[s].items():
                    for i2 in range(2):
                        for j2 in range(2):
                            i = br * 2 + i2
                            j = bc * 2 + j2
                            U[i * 4 + j, c] += coeff
                for (tag, br, bc), coeff in Vforms[s].items():
                    for j2 in range(2):
                        for k2 in range(2):
                            j = br * 2 + j2
                            k = bc * 2 + k2
                            V[j * 4 + k, c] += coeff
                for (tag, br, bc), coeff in Wforms[s].items():
                    for k2 in range(2):
                        for i2 in range(2):
                            k = br * 2 + k2
                            i = bc * 2 + i2
                            W[k * 4 + i, c] += coeff

    assert _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4), "4x4x4 Strassen broken"
    return U, V, W


def _build_4x4x5():
    """Combine 4x4x4 Strassen (rank 28) + boundary 4x4 x 4x1 (rank 16) = rank 44.
    U:(16,44) V:(20,44) W:(20,44)."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_total = 44
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))

    U[:, :rank_core] = U44
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :rank_core] = V44[j * 4 + k, :rank_core]
    W[:16, :rank_core] = W44

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0  # k=4, i

    assert _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P), "4x4x5 broken"
    return U, V, W


def _is_col_dependent(col, other_cols, tol=1e-8):
    """Fast check: is `col` in the column space of `other_cols`?
    Uses QR with pivoting on [other_cols | col] and checks if the last
    diagonal is near zero (meaning col adds no new dimension)."""
    if other_cols.shape[1] == 0:
        return np.allclose(col, 0, atol=tol)
    combined = np.column_stack([other_cols, col])
    Q, R = np.linalg.qr(combined, mode='complete')
    # The rank of other_cols vs combined: if col is dependent,
    # the last column of R (corresponding to col) should be ~0 in its
    # diagonal position relative to the rank of other_cols.
    # Simpler: compare norms of R diagonal
    diag = np.abs(np.diag(R))
    # Find effective rank of other_cols
    rank_oc = np.sum(diag[:-1] > tol * diag[0] if len(diag) > 1 else diag[0] > tol)
    # If col is dependent, the (rank_oc, rank_oc) entry of R (0-indexed)
    # which corresponds to the last column's contribution should be small
    if rank_oc < other_cols.shape[1]:
        # other_cols itself is rank-deficient; just check if col adds rank
        return np.linalg.matrix_rank(combined, tol=tol) == np.linalg.matrix_rank(other_cols, tol=tol)
    # Check if the last column's R-diagonal is small
    return diag[-1] < tol * max(diag[0], 1.0)


def _reduce_columns(U, V, W, tol=1e-9, max_iter=400, deadline=None):
    """Remove columns that are exactly zero or make decomposition still valid.
    Uses QR-based linear-dependency pruning before the exact verify."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    removed_any = False
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        for c in range(rank - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            # Quick prune: check if column c is a linear combo of others in U
            Uc = np.array([row[c] for row in U])
            Uo = np.array([row[:c] + row[c + 1:] for row in U])
            if Uo.shape[1] > 0:
                if _is_col_dependent(Uc, Uo, tol=1e-8):
                    continue
            # Also check V and W for quick prune (any one being independent means
            # the column might still be needed, but if all three show dependency
            # it's very likely removable)
            Vc = np.array([row[c] for row in V])
            Vo = np.array([row[:c] + row[c + 1:] for row in V])
            Wc = np.array([row[c] for row in W])
            Wo = np.array([row[:c] + row[c + 1:] for row in W])
            dep_U = _is_col_dependent(Uc, Uo, tol=1e-8) if Uo.shape[1] > 0 else np.allclose(Uc, 0, atol=1e-8)
            dep_V = _is_col_dependent(Vc, Vo, tol=1e-8) if Vo.shape[1] > 0 else np.allclose(Vc, 0, atol=1e-8)
            dep_W = _is_col_dependent(Wc, Wo, tol=1e-8) if Wo.shape[1] > 0 else np.allclose(Wc, 0, atol=1e-8)
            # Only do expensive verify if at least one factor shows the column is independent
            # (i.e., not all three show dependency — if all show dependency, high chance removable)
            if not (dep_U or dep_V or dep_W):
                # All three factors show column is dependent — very likely removable
                pass  # fall through to verify
            # Actually, let's just verify if any factor shows independence
            # (the column might be needed even if one factor has it dependent)
            # Better: just always verify if we get here (not pruned by U)
            Ut = [row[:c] + row[c + 1:] for row in U]
            Vt = [row[:c] + row[c + 1:] for row in V]
            Wt = [row[:c] + row[c + 1:] for row in W]
            if _verify(Ut, Vt, Wt, N, M, P, tol):
                U, V, W = Ut, Vt, Wt
                rank -= 1
                removed = True
                removed_any = True
                break
        if not removed:
            break
    if removed_any:
        return U, V, W
    return None


def _sparse_random_search(U, V, W, budget, tol=1e-9):
    """Random sparse perturbation + column removal.
    Perturb a random subset of entries with integer values (±1),
    then attempt column removal. Keep any strictly better valid decomposition."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    best_rank = rank
    best = (U, V, W)
    start = time.time()
    factors = [U, V, W]
    ntries = 0
    while time.time() - start < budget:
        ntries += 1
        f = random.randrange(3)
        F = factors[f]
        rrow = random.randrange(len(F))
        ncol = len(F[0])
        # Perturb with integer values to maintain integer/half-integer constraint
        for _ in range(random.randrange(1, 4)):
            c2 = random.randrange(ncol)
            F[rrow][c2] += random.choice([-1.0, 1.0])
        # Also try perturbing in other factors simultaneously for more mixing
        if random.random() < 0.5:
            f2 = random.randrange(3)
            if f2 != f:
                F2 = factors[f2]
                rrow2 = random.randrange(len(F2))
                for _ in range(random.randrange(1, 3)):
                    c2 = random.randrange(len(F2[0]))
                    F2[rrow2][c2] += random.choice([-1.0, 1.0])
        red = _reduce_columns(factors[0], factors[1], factors[2],
                              tol=tol, max_iter=80,
                              deadline=start + budget)
        if red is not None:
            Ur, Vr, Wr = red
            rr = len(Ur[0])
            if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P, tol):
                best_rank = rr
                best = (Ur, Vr, Wr)
                U, V, W = Ur, Vr, Wr
                rank = rr
                factors = [U, V, W]
        # Reset to best if we went off-track
        if time.time() - start < budget:
            # Keep working from current state (which is best so far)
            pass
    return best


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # fallback: schoolbook
    Us, Vs, Ws = _schoolbook(N, M, P)
    assert _verify(Us, Vs, Ws, N, M, P)
    best_rank = N * M * P
    best = {"U": Us, "V": Vs, "W": Ws}

    # structured 4x4x5 (rank 44)
    if time.time() < deadline:
        try:
            U445, V445, W445 = _build_4x4x5()
            if _verify(U445.tolist(), V445.tolist(), W445.tolist(), N, M, P):
                best_rank = 44
                best = {"U": U445.tolist(), "V": V445.tolist(), "W": W445.tolist()}

                # reduce via column removal
                if time.time() < deadline:
                    red = _reduce_columns(best["U"], best["V"], best["W"],
                                          max_iter=300,
                                          deadline=deadline - 10.0)
                    if red is not None:
                        Ur, Vr, Wr = red
                        rr = len(Ur[0])
                        if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                            best_rank = rr
                            best = {"U": Ur, "V": Vr, "W": Wr}

                    # random sparse perturbation search to shave more columns
                    if time.time() < deadline:
                        remain = deadline - time.time() - 5.0
                        if remain > 3.0:
                            Ur, Vr, Wr = _sparse_random_search(
                                best["U"], best["V"], best["W"], budget=remain)
                            rr = len(Ur[0])
                            if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                                best_rank = rr
                                best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    return best
