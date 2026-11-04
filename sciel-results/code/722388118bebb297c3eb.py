"""
Rank-minimal tensor decomposition for 2x5x6 matrix multiplication.

Strategy: Block decomposition with optimized sub-blocks.
- Use Strassen (rank 7) for 2x2x2 blocks
- Use schoolbook (rank 4) for 2x1x2 blocks

Block layout for 2x5x6:
  m=5 split as [2, 2, 1], p=6 split as [2, 2, 2]
  Blocks:
    (j=0..1, k=0..1): 2x2x2 Strassen, rank 7
    (j=0..1, k=2..3): 2x2x2 Strassen, rank 7
    (j=0..1, k=4..5): 2x2x2 Strassen, rank 7
    (j=2..3, k=0..1): 2x2x2 Strassen, rank 7
    (j=2..3, k=2..3): 2x2x2 Strassen, rank 7
    (j=2..3, k=4..5): 2x2x2 Strassen, rank 7
    (j=4, k=0..1):    2x1x2 schoolbook, rank 4
    (j=4, k=2..3):    2x1x2 schoolbook, rank 4
    (j=4, k=4..5):    2x1x2 schoolbook, rank 4
Total: 6*7 + 3*4 = 54

We also attempt to reduce the rank via linear algebra techniques.
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 135.0  # Leave 15s safety margin for 150s limit


def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    result = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(result, target, atol=tol)


def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def strassen_decomposition_2x2():
    """
    Returns Strassen decomposition for a 2x2x2 block.
    U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank_block = 7
    U_block = np.zeros((4, rank_block))
    V_block = np.zeros((4, rank_block))
    W_block = np.zeros((4, rank_block))

    # Term 0: M1 = (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1

    # Term 1: M2 = (A10+A11)B00
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1

    # Term 2: M3 = A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1

    # Term 3: M4 = A11(B10-B00)
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1

    # Term 4: M5 = (A00+A10)B11
    U_block[0, 4] = 1; U_block[2, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1

    # Term 5: M6 = (A10-A00)(B00+B01)
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1

    # Term 6: M7 = (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1

    return U_block, V_block, W_block


def schoolbook_2x1x2():
    """
    Returns schoolbook decomposition for a 2x1x2 block.
    n=2, m=1, p=2. Rank = 4.
    U_block: (2, 4), V_block: (2, 4), W_block: (4, 4)
    """
    rank_block = 4
    U_block = np.zeros((2, rank_block))
    V_block = np.zeros((2, rank_block))
    W_block = np.zeros((4, rank_block))

    # c=0: i=0, j=0, k=0
    U_block[0, 0] = 1
    V_block[0, 0] = 1
    W_block[0, 0] = 1

    # c=1: i=0, j=0, k=1
    U_block[0, 1] = 1
    V_block[1, 1] = 1
    W_block[2, 1] = 1

    # c=2: i=1, j=0, k=0
    U_block[1, 2] = 1
    V_block[0, 2] = 1
    W_block[1, 2] = 1

    # c=3: i=1, j=0, k=1
    U_block[1, 3] = 1
    V_block[1, 3] = 1
    W_block[3, 3] = 1

    return U_block, V_block, W_block


def build_block_decomposition(n, m, p):
    """
    Builds a decomposition using multiple block types.
    Layout: m=5 as [2,2,1], p=6 as [2,2,2]
    Total rank: 6*7 + 3*4 = 54.
    """
    max_rank = 60
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))

    col = 0

    blocks = [
        (0, 0, 2, 2, 'strassen'),
        (0, 2, 2, 2, 'strassen'),
        (0, 4, 2, 2, 'strassen'),
        (2, 0, 2, 2, 'strassen'),
        (2, 2, 2, 2, 'strassen'),
        (2, 4, 2, 2, 'strassen'),
        (4, 0, 1, 2, 'schoolbook_2x1x2'),
        (4, 2, 1, 2, 'schoolbook_2x1x2'),
        (4, 4, 1, 2, 'schoolbook_2x1x2'),
    ]

    for j0, k0, mj, pk, method in blocks:
        if method == 'strassen':
            Ub, Vb, Wb = strassen_decomposition_2x2()
            for i in range(2):
                for j in range(mj):
                    row_idx = i*m + (j0+j)
                    U[row_idx, col:col+7] = Ub[i*2+j, :]
            for j in range(mj):
                for k in range(pk):
                    row_idx = (j0+j)*p + (k0+k)
                    V[row_idx, col:col+7] = Vb[j*2+k, :]
            for k in range(pk):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+7] = Wb[k*2+i, :]
            col += 7
        elif method == 'schoolbook_2x1x2':
            Ub, Vb, Wb = schoolbook_2x1x2()
            for i in range(2):
                row_idx = i*m + j0
                U[row_idx, col:col+4] = Ub[i, :]
            for k in range(pk):
                row_idx = j0*p + (k0+k)
                V[row_idx, col:col+4] = Vb[k, :]
            for k in range(pk):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+4] = Wb[k*2+i, :]
            col += 4

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    if not check_decomposition(U, V, W, n, m, p):
        raise ValueError("Block decomposition failed verification")

    return U, V, W


def try_column_reduction(U, V, W, n, m, p):
    """
    Attempt to reduce the rank by finding linear dependencies.
    Uses faster verification with precomputed tensors.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Precompute the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_fast(Uc, Vc, Wc):
        """Fast check using einsum."""
        result = np.einsum('ac,bc,dc->abd', Uc, Vc, Wc)
        return np.allclose(result, target, atol=1e-9)

    for c in range(rank):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        U_test = np.delete(U, c, axis=1)
        V_test = np.delete(V, c, axis=1)
        W_test = np.delete(W, c, axis=1)
        if check_fast(U_test, V_test, W_test):
            U, V, W = U_test, V_test, W_test
            rank -= 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            c = -1  # Restart

    return best_U, best_V, best_W, best_rank


def try_linear_combination_removal(U, V, W, n, m, p):
    """
    Try to find if any column is a linear combination of others.
    Uses faster verification.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_fast(Uc, Vc, Wc):
        result = np.einsum('ac,bc,dc->abd', Uc, Vc, Wc)
        return np.allclose(result, target, atol=1e-9)

    for c in range(rank):
        if time.time() - START_TIME > TIME_LIMIT:
            break

        other_cols = [d for d in range(rank) if d != c]
        if len(other_cols) < 3:
            continue

        U_others = U[:, other_cols]
        V_others = V[:, other_cols]
        W_others = W[:, other_cols]

        U_c = U[:, c]
        V_c = V[:, c]
        W_c = W[:, c]

        try:
            alpha_U, _, _, _ = np.linalg.lstsq(U_others, U_c, rcond=None)
            residual_U = np.linalg.norm(U_c - U_others @ alpha_U)
        except:
            continue

        if residual_U > 1e-8:
            continue

        try:
            V_pred = V_others @ alpha_U
            residual_V = np.linalg.norm(V_c - V_pred)
        except:
            continue

        if residual_V > 1e-8:
            continue

        try:
            W_pred = W_others @ alpha_U
            residual_W = np.linalg.norm(W_c - W_pred)
        except:
            continue

        if residual_W > 1e-8:
            continue

        U = np.delete(U, c, axis=1)
        V = np.delete(V, c, axis=1)
        W = np.delete(W, c, axis=1)
        rank -= 1
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        c = -1  # Restart

    return best_U, best_V, best_W, best_rank


def try_kronecker_reduction(U, V, W, n, m, p):
    """
    Try to exploit structure to reduce rank.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_fast(Uc, Vc, Wc):
        result = np.einsum('ac,bc,dc->abd', Uc, Vc, Wc)
        return np.allclose(result, target, atol=1e-9)

    # Try removing pairs of columns that might be dependent
    for c1 in range(rank - 1):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            U_cols = np.column_stack([U[:, c1], U[:, c2]])
            V_cols = np.column_stack([V[:, c1], V[:, c2]])
            W_cols = np.column_stack([W[:, c1], W[:, c2]])
            
            if np.linalg.matrix_rank(U_cols) == 1 and \
               np.linalg.matrix_rank(V_cols) == 1 and \
               np.linalg.matrix_rank(W_cols) == 1:
                U = np.delete(U, c2, axis=1)
                V = np.delete(V, c2, axis=1)
                W = np.delete(W, c2, axis=1)
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Restart inner loop
    
    return best_U, best_V, best_W, best_rank


def construct():
    global START_TIME
    n, m, p = DIMENSIONS

    # 1. Build block decomposition (rank 54)
    try:
        U, V, W = build_block_decomposition(n, m, p)
    except:
        U, V, W = schoolbook_decomposition(n, m, p)

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # 2. Try Kronecker-structure-aware reduction
    if time.time() - START_TIME < TIME_LIMIT - 40:
        U2, V2, W2, rank2 = try_kronecker_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2

    # 3. Try linear combination removal
    if time.time() - START_TIME < TIME_LIMIT - 25:
        U3, V3, W3, rank3 = try_linear_combination_removal(best_U, best_V, best_W, n, m, p)
        if rank3 < best_rank:
            best_rank = rank3
            best_U, best_V, best_W = U3, V3, W3

    # 4. Try simple column removal
    if time.time() - START_TIME < TIME_LIMIT - 15:
        U4, V4, W4, rank4 = try_column_reduction(best_U, best_V, best_W, n, m, p)
        if rank4 < best_rank:
            best_rank = rank4
            best_U, best_V, best_W = U4, V4, W4

    # 5. Final verification
    if not check_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = schoolbook_decomposition(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
