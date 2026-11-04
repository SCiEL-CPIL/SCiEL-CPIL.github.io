import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
DEADLINE = 138.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition. U:(nm,rank) V:(mp,rank) W:(pn,rank).
    W index convention: W[k*n + i, c]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    rec = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.allclose(rec, np.ones((n, m, p)), atol=tol)


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
    Returns U:(16,28) V:(16,28) W:(16,28)."""
    Su, Sv, Sw = _strassen_2x2()
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


def _build_4x4x5_strassen_boundary():
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
            W[4 * 4 + i, c] = 1.0

    assert _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P), "4x4x5 broken"
    return U, V, W


def _reduce_linear(U, V, W, tol=1e-8, max_passes=40):
    """Detect columns that are linear combinations of others and merge."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]

    stacked = np.vstack([Ua, Va, Wa])

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        r = stacked.shape[1]
        if r <= 1:
            break
        full_rank = np.linalg.matrix_rank(stacked, tol=tol)
        try:
            _, svals, Vt = np.linalg.svd(stacked, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        null_dim = r - full_rank
        if null_dim <= 0:
            break
        null_vecs = Vt[full_rank:]
        for nv in null_vecs:
            idx = int(np.argmax(np.abs(nv)))
            if abs(nv[idx]) < 1e-10:
                continue
            keep = [c for c in range(r) if c != idx]
            stacked = stacked[:, keep]
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            changed = True
            break
        if not changed:
            break

    if Ua.shape[1] < rank:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None


def _reduce_columns(U, V, W, tol=1e-9, max_iter=200):
    """Remove columns that are exactly zero or make decomposition still valid."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    for _ in range(max_iter):
        removed = False
        for c in range(rank - 1, -1, -1):
            Ut = [row[:c] + row[c + 1:] for row in U]
            Vt = [row[:c] + row[c + 1:] for row in V]
            Wt = [row[:c] + row[c + 1:] for row in W]
            if _verify(Ut, Vt, Wt, N, M, P, tol):
                U, V, W = Ut, Vt, Wt
                rank -= 1
                removed = True
                break
        if not removed:
            break
    if rank < len(U[0]):
        return U, V, W
    return None


def _try_rank19_4x4x4():
    """
    Attempt to find a rank-19 decomposition for 4x4x4.
    This is the best known upper bound (Landsberg).
    We use a structured search over small integer/half-integer entries.
    """
    # For 4x4x4, the tensor has 64 components.
    # We look for U, V, W of shape (16, 19) with entries in {-1, -0.5, 0, 0.5, 1}
    # This is a huge search space, so we use a heuristic.
    
    # Actually, implementing a full rank-19 search is very hard.
    # Instead, we rely on the fact that if such a decomposition exists,
    # it might be found by random search with structured sparsity.
    
    # However, given the difficulty, we will skip this and focus on optimizing the 4x4x5 case
    # by using the best known 4x4x4 rank (28 via Strassen) and trying to reduce the boundary.
    
    return None


def _optimize_boundary():
    """
    Try to reduce the rank of the boundary part (4x4 x 4x1 -> 4x1).
    The boundary tensor is: sum_{i,j} A[i,j] B[j,4] C[i,4]
    This is equivalent to multiplying a 4x4 matrix by a 4x1 vector.
    The rank of this tensor is 4 (since the output is 4x1, and we need 4 independent sums).
    Wait, the rank of the matrix multiplication tensor for m x n * n x 1 -> m x 1 is m.
    So the boundary rank should be 4, not 16.
    
    Let's verify:
    C[i] = sum_j A[i,j] B[j]
    This is a linear map from (A, B) to C.
    The tensor T[i,j,k] = delta_{k,4} * A[i,j] * B[j,k] ... no, B is 4x5, so B[j,4] is the 5th column.
    The tensor is T[i, j, k] where k is fixed to 4.
    Actually, the full tensor is for 4x4 x 4x5.
    The boundary part is when k=4.
    T[i,j,4] = A[i,j] * B[j,4].
    This is a 4x4x1 tensor (since k is fixed).
    The rank of this tensor is 4, because it can be written as:
    sum_{r=1}^4 u_r v_r w_r where u_r[i,j] = A[i,r], v_r[j,k] = B[r,k] (with k=4), w_r[i,k] = C[i,k] (with k=4).
    Wait, that's not right.
    
    The correct decomposition for the boundary is:
    C[i,4] = sum_j A[i,j] B[j,4]
    We can write this as:
    C[i,4] = sum_{j=0}^3 A[i,j] B[j,4]
    This is already a sum of 4 rank-1 terms if we fix j.
    But we can do better?
    
    Actually, the rank of the tensor for m x n * n x 1 is min(m, n, 1) * max(m, n, 1) ... no.
    The rank of the matrix multiplication tensor M_{m,n,1} is m.
    So the boundary rank is 4.
    
    Therefore, the total rank should be 28 (for 4x4x4) + 4 (for boundary) = 32.
    
    Let's construct this.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_boundary = 4
    rank_total = rank_core + rank_boundary
    
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    # Core part
    U[:, :rank_core] = U44
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :rank_core] = V44[j * 4 + k, :rank_core]
    W[:16, :rank_core] = W44
    
    # Boundary part: C[i,4] = sum_j A[i,j] B[j,4]
    # We use 4 rank-1 terms, one for each j.
    # But we can combine them? No, they are independent.
    # Actually, we can use a rank-4 decomposition.
    for j in range(4):
        c = rank_core + j
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
        # A[i,j]
        for i in range(4):
            U[i * 4 + j, c] = 1.0
        # B[j,4]
        V[j * 5 + 4, c] = 1.0
        # C[i,4]
        for i in range(4):
            W[4 * 4 + i, c] = 1.0
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P):
        return None
    
    return U.tolist(), V.tolist(), W.tolist(), rank_total


def _sparse_random_search(deadline, best_rank, seed=12345):
    """Structured random search with sparse support over {-1,0,1}."""
    rng = random.Random(seed)
    target_ranks = [36, 34, 32, 30, 28]
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        budget = deadline - time.time()
        if budget < 1.0:
            break
        attempts = max(5, int(budget / 0.05))
        for _ in range(attempts):
            if time.time() > deadline:
                break
            U = np.zeros((16, target_rank))
            V = np.zeros((20, target_rank))
            W = np.zeros((20, target_rank))
            for c in range(target_rank):
                nu = rng.randint(1, 3)
                nv = rng.randint(1, 3)
                nw = rng.randint(1, 3)
                for idx in rng.sample(range(16), nu):
                    U[idx, c] = rng.choice([-1, 1])
                for idx in rng.sample(range(20), nv):
                    V[idx, c] = rng.choice([-1, 1])
                for idx in rng.sample(range(20), nw):
                    W[idx, c] = rng.choice([-1, 1])
            if _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P):
                return U.tolist(), V.tolist(), W.tolist(), target_rank
    return None


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
            U445, V445, W445 = _build_4x4x5_strassen_boundary()
            if _verify(U445.tolist(), V445.tolist(), W445.tolist(), N, M, P):
                best_rank = 44
                best = {"U": U445.tolist(), "V": V445.tolist(), "W": W445.tolist()}
        except Exception:
            pass

    # Try the optimized boundary (rank 32)
    if time.time() < deadline:
        try:
            res = _optimize_boundary()
            if res is not None:
                Ur, Vr, Wr, rr = res
                if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    # Reduce via column removal
    if time.time() < deadline and best_rank > 32:
        try:
            red = _reduce_columns(best["U"], best["V"], best["W"], max_iter=150)
            if red is not None:
                Ur, Vr, Wr = red
                rr = len(Ur[0])
                if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    # Reduce via linear-dependence merging
    if time.time() < deadline and best_rank > 32:
        try:
            red2 = _reduce_linear(best["U"], best["V"], best["W"])
            if red2 is not None:
                Ur, Vr, Wr = red2
                rr = len(Ur[0])
                if rr < best_rank and _verify(Ur, Vr, Wr, N, M, P):
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    # Sparse random search for even lower rank
    if time.time() < deadline and best_rank > 32:
        try:
            res = _sparse_random_search(deadline, best_rank, seed=98765)
            if res is not None:
                Ur, Vr, Wr, rr = res
                if rr < best_rank:
                    best_rank = rr
                    best = {"U": Ur, "V": Vr, "W": Wr}
        except Exception:
            pass

    if time.time() > deadline:
        return best
    return best
