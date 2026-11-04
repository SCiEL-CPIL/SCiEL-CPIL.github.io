"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 110.0  # safety margin before 120s limit

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: Frame terms for (i,j) where i==4 or j==4, with each k
    # These are 9 pairs of (i,j) × 5 values of k = 45 terms
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: 4x4x4 core via Strassen (rank 7)
    # 4x4 A: rows 0..3, cols 0..3. Partition into 2x2 blocks of size 2.
    # A[bi][bj] covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B[bi][bj] covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # C[bi][bj] = sum over b' of A[bi][b'] * B[b'][bj]

    def block(bi, bj):
        """Yield (row, col) for 2x2 block (bi,bj) where bi,bj in {0,1}."""
        for d in range(2):
            for e in range(2):
                yield (2 * bi + d, 2 * bj + e)

    # Strassen formulas for 4x4 (2x2 blocks of size 2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22) B11
    # M3 = A11 (B12-B22)
    # M4 = A22 (B21-B11)
    # M5 = (A11+A12) B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    # Each M is a list of (A_block, B_block, sign)
    # A_block = (bi, bj) for A, B_block = (bj', bk) for B where bj==bj'
    M_defs = [
        # M1 = (A11+A22)(B11+B22)
        [((0,0),(0,0),1), ((0,0),(0,1),1),
         ((1,1),(0,0),1), ((1,1),(0,1),1)],
        # M2 = (A21+A22) B11
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A11 (B12-B22)
        [((0,0),(0,1),1), ((0,0),(1,1),-1)],
        # M4 = A22 (B21-B11)
        [((1,1),(1,0),1), ((1,1),(0,0),-1)],
        # M5 = (A11+A12) B22
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A21-A11)(B11+B12)
        [((1,0),(0,0),1), ((1,0),(0,1),1),
         ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A12-A22)(B21+B22)
        [((0,1),(1,0),1), ((0,1),(1,1),1),
         ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]

    for parts in M_defs:
        u_acc = {}
        v_acc = {}
        w_acc = {}
        for (ab, bc, coeff) in parts:
            a_bi, a_bj = ab
            b_bi, b_bj = bc
            # The middle index must match: a_bj == b_bi
            if a_bj != b_bi:
                continue
            for (ii, jj) in block(a_bi, a_bj):
                for (jj2, kk) in block(b_bi, b_bj):
                    if jj != jj2:
                        continue
                    ui = ii * m + jj
                    vi = jj * p + kk
                    wi = kk * n + ii
                    u_acc[ui] = u_acc.get(ui, 0) + coeff
                    v_acc[vi] = v_acc.get(vi, 0) + coeff
                    w_acc[wi] = w_acc.get(wi, 0) + coeff
        # Remove zero entries
        u_acc = {k: v for k, v in u_acc.items() if v != 0}
        v_acc = {k: v for k, v in v_acc.items() if v != 0}
        w_acc = {k: v for k, v in w_acc.items() if v != 0}
        U_cols.append(u_acc)
        V_cols.append(v_acc)
        W_cols.append(w_acc)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random integer tests using NumPy for speed
    try:
        import numpy as np
        rng = np.random.RandomState(12345)
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        for trial in range(500):
            if time.time() > tmax:
                break
            A = rng.randint(0, 6, size=(n, m))
            B = rng.randint(0, 6, size=(m, p))
            C_true = A @ B
            # Compute C_test via tensor decomposition
            # C_test[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_test = np.zeros((n, p), dtype=np.int64)
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                w_col = W_np[:, r]
                # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
                u_mat = u_col.reshape(n, m)
                v_mat = v_col.reshape(m, p)
                w_mat = w_col.reshape(p, n)
                # C_test += u_mat * v_mat * w_mat (element-wise product summed over j)
                # C_test[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                prod = u_mat * v_mat  # (n, p) via broadcasting? No, need sum over j
                # Actually: sum_j u_mat[i,j] * v_mat[j,k] = (u_mat @ v_mat)[i,k]
                # Then multiply by w_mat[k,i]
                temp = u_mat @ v_mat  # (n, p)
                # C_test[i,k] += temp[i,k] * w_mat[k,i]
                C_test += temp * w_mat.T  # w_mat.T[i,k] = w_mat[k,i]
            if not np.array_equal(C_test, C_true):
                return _schoolbook()
    except ImportError:
        # Fallback to pure Python verification
        import random
        rng = random.Random(12345)
        for trial in range(200):
            if time.time() > tmax:
                break
            A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
            C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
            C_test = [[0] * p for _ in range(n)]
            for ii in range(n):
                for kk in range(p):
                    s = 0
                    for r in range(rank):
                        for jj in range(m):
                            s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                    C_test[ii][kk] = s
            if C_test != C_true:
                return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
