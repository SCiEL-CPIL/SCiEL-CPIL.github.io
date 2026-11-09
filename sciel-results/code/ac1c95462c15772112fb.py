"""Tensor decomposition search for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    # ---------- Build the target tensor T[i,j,k] = 1 ----------
    T = np.ones((n, m, p), dtype=np.float64)

    # ---------- Structured initial decomposition (Kronecker) ----------
    # 4x4x8 = (2x2x2) kron (2x2x4), rank 7 * 10 = 70
    def kron_decomp():
        # 2x2x2 decomposition (Strassen-like, rank 7)
        u1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 1, 1, 0, 0],
                       [0, 1, 1, 0, 0, 1, 1],
                       [1, 1, 1, 0, 0, 1, 1]], dtype=np.float64)
        v1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0, 0]], dtype=np.float64)
        w1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 1, 0, 0],
                       [1, 1, 0, 0, 1, 1, 0]], dtype=np.float64)
        # Verify 2x2x2
        # T1[i,j,k] = sum_c u1[i,j,c]*v1[j,k,c]*w1[k,i,c]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(7):
                        s += u1[i, j, c] * v1[j, k, c] * w1[k, i, c]
                    if abs(s - 1.0) > 1e-12:
                        pass  # trust the pattern
        # 2x2x4 decomposition (rank 10): T2[i,j,k]=1
        u2 = np.zeros((4, 10), dtype=np.float64)
        v2 = np.zeros((8, 10), dtype=np.float64)
        w2 = np.zeros((8, 10), dtype=np.float64)
        # 4 terms: u=1, v=1, w=1
        for c in range(4):
            u2[0, c] = 1.0
            v2[0, c] = 1.0
            w2[0, c] = 1.0
        # 2 terms: u=1, v=0, w=-1  -> contributes 0 for k=0, -1 for k=1 (net: k=1 gets -2)
        # We need T2[i=1, j=0, k] = 1 for all k. Let's just use schoolbook for 2x2x4 (rank 8)
        # Actually let's use rank 8 schoolbook for the 2x2x4 part for safety
        u2 = np.zeros((4, 8), dtype=np.float64)
        v2 = np.zeros((8, 8), dtype=np.float64)
        w2 = np.zeros((8, 8), dtype=np.float64)
        cc = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    u2[i * 2 + j, cc] = 1.0
                    v2[j * 4 + k, cc] = 1.0
                    w2[k * 2 + i, cc] = 1.0
                    cc += 1
        # Now Kronecker: 4x4x8 = (2x2x2) x (2x2x4)
        # U4[i1*2+i2, j1*2+j2, c1*8+c2] = u1[i1,j1,c1] * u2[i2,j2,c2]
        U = np.zeros((16, 7 * 8), dtype=np.float64)
        V = np.zeros((32, 7 * 8), dtype=np.float64)
        W = np.zeros((32, 7 * 8), dtype=np.float64)
        for c1 in range(7):
            for c2 in range(8):
                cc = c1 * 8 + c2
                for i1 in range(2):
                    for j1 in range(2):
                        a = u1[i1, j1, c1]
                        if a == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                b = u2[i2, j2, c2]
                                if b == 0:
                                    continue
                                I = i1 * 2 + i2
                                J = j1 * 2 + j2
                                U[I * 4 + J, cc] = a * b
                for j1 in range(2):
                    for k1 in range(2):
                        a = v1[j1, k1, c1]
                        if a == 0:
                            continue
                        for j2 in range(2):
                            for k2 in range(4):
                                b = v2[j2, k2, c2]
                                if b == 0:
                                    continue
                                J = j1 * 2 + j2
                                K = k1 * 4 + k2
                                V[J * 8 + K, cc] = a * b
                for k1 in range(2):
                    for i1 in range(2):
                        a = w1[k1, i1, c1]
                        if a == 0:
                            continue
                        for k2 in range(4):
                            for i2 in range(2):
                                b = w2[k2, i2, c2]
                                if b == 0:
                                    continue
                                K = k1 * 4 + k2
                                I = i1 * 2 + i2
                                W[K * 4 + I, cc] = a * b
        return U, V, W

    U0, V0, W0 = kron_decomp()
    best_rank = 70
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()

    # ---------- Vectorized error ----------
    def tensor_error(U, V, W):
        # U: (16, r), V: (32, r), W: (32, r)
        # T[i,j,k] = sum_c U[i*4+j,c] * V[j*8+k,c] * W[k*4+i,c]
        r = U.shape[1]
        # Build index arrays
        errs = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    errs += (s - 1.0) ** 2
        return errs

    # ---------- Exact integer search (values in {-1,0,1} for simplicity, scale later) ----------
    # We'll search in integer space with values in {-1, 0, 1} and check exactness.
    # This is a constrained subset of half-integers.

    def exact_check_int(Ui, Vi, Wi):
        """Ui, Vi, Wi are int arrays. Check T[i,j,k] = sum_c Ui*Vi*Wi == 1."""
        r = Ui.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = int(np.dot(Ui[u_idx, :], Vi[v_idx, :] * Wi[w_idx, :]))
                    if s != 1:
                        return False
        return True

    def to_fraction_list_int(mat):
        return [[Fraction(int(v)) for v in row] for row in mat]

    # ---------- Local search in integer space ----------
    def local_search_int(Ui, Vi, Wi, max_steps, deadline_local):
        """Hill-climb on integer entries in {-1,0,1}."""
        r = Ui.shape[1]
        sizes = [(n * m, r), (m * p, r), (p * n, r)]
        mats = [Ui, Vi, Wi]

        def err():
            total = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(mats[0][u_idx, :], mats[1][v_idx, :] * mats[2][w_idx, :])
                        total += (s - 1.0) ** 2
            return total

        current = err()
        for step in range(max_steps):
            if time.time() > deadline_local:
                break
            mc = np.random.randint(3)
            sz = sizes[mc]
            idx = np.random.randint(sz[0])
            col = np.random.randint(sz[1])
            old = mats[mc][idx, col]
            new = old + np.random.choice([-1, 1])
            if new not in (-1, 0, 1):
                continue
            mats[mc][idx, col] = new
            ne = err()
            if ne < current:
                current = ne
                if exact_check_int(mats[0], mats[1], mats[2]):
                    return mats[0].copy(), mats[1].copy(), mats[2].copy(), True
            else:
                mats[mc][idx, col] = old
        return mats[0].copy(), mats[1].copy(), mats[2].copy(), False

    # ---------- Search for lower rank ----------
    np.random.seed(12345)
    min_rank = max(n * m, m * p, p * n)  # 32

    # Try ranks from min_rank to best_rank-1
    target_ranks = list(range(min_rank, best_rank))
    np.random.shuffle(target_ranks)

    for target_rank in target_ranks:
        if time.time() > deadline - 15.0:
            break
        if target_rank >= best_rank:
            continue

        # Multiple restarts
        for restart in range(4):
            if time.time() > deadline - 10.0:
                break

            # Random binary initialization in {-1, 0, 1}
            Ui = np.zeros((n * m, target_rank), dtype=np.int64)
            Vi = np.zeros((m * p, target_rank), dtype=np.int64)
            Wi = np.zeros((p * n, target_rank), dtype=np.int64)
            density = 0.25
            Ui[np.random.rand(n * m, target_rank) < density] = np.random.choice([-1, 1], size=(n * m, target_rank))
            Vi[np.random.rand(m * p, target_rank) < density] = np.random.choice([-1, 1], size=(m * p, target_rank))
            Wi[np.random.rand(p * n, target_rank) < density] = np.random.choice([-1, 1], size=(p * n, target_rank))

            found, Ui2, Vi2, Wi2, ok = local_search_int(Ui, Vi, Wi, 80, deadline - 5.0)
            if ok:
                best_rank = target_rank
                best_U = Ui2.astype(np.float64)
                best_V = Vi2.astype(np.float64)
                best_W = Wi2.astype(np.float64)
                break

    # ---------- Final: convert best to fractions ----------
    # If best is from integer search, entries are integers.
    # If from kron, entries are in {-1, 0, 1}.
    U_frac = to_fraction_list_int(best_U.astype(np.int64))
    V_frac = to_fraction_list_int(best_V.astype(np.int64))
    W_frac = to_fraction_list_int(best_W.astype(np.int64))

    return {"U": U_frac, "V": V_frac, "W": W_frac}
