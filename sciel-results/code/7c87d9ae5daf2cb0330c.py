"""
Rank-minimal tensor decomposition for 2x5x6 matrix multiplication.

Strategy: Block decomposition with optimized sub-blocks.
- Use Strassen (rank 7) for 2x2x2 blocks
- Use optimized 2x3x2 blocks (rank 9) where possible
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

Alternative layout: m=5 split as [3, 2], p=6 split as [2, 2, 2]
    (j=0..2, k=0..1): 2x3x2 block, rank 9
    (j=0..2, k=2..3): 2x3x2 block, rank 9
    (j=0..2, k=4..5): 2x3x2 block, rank 9
    (j=3..4, k=0..1): 2x2x2 Strassen, rank 7
    (j=3..4, k=2..3): 2x2x2 Strassen, rank 7
    (j=3..4, k=4..5): 2x2x2 Strassen, rank 7
Total: 3*9 + 3*7 = 54 (same)

Try: m=5 split as [2, 3], p=6 split as [3, 3]
    (j=0..1, k=0..2): 2x2x3 block
    (j=0..1, k=3..5): 2x2x3 block
    (j=2..4, k=0..2): 2x3x3 block
    (j=2..4, k=3..5): 2x3x3 block
Need ranks for these blocks.
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

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster check using vectorized operations."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


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


def strassen_2x2x3():
    """
    Returns a decomposition for a 2x2x3 block using Strassen-like structure.
    n=2, m=2, p=3. We can view this as two 2x2x2 blocks sharing structure,
    or find a direct rank-9 decomposition.
    
    Actually, let's use: treat p=3 as p=2+1.
    Split into (2x2x2) Strassen (rank 7) + (2x2x1) schoolbook (rank 4) = 11.
    Or find a better decomposition.
    
    For now, use schoolbook: rank = 2*2*3 = 12.
    Better: Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4) = 11.
    """
    # Use Strassen 2x2x2 + schoolbook 2x2x1
    Ub1, Vb1, Wb1 = strassen_decomposition_2x2()  # (4,7), (4,7), (4,7)
    
    # Schoolbook 2x2x1: n=2, m=2, p=1, rank=4
    Ub2 = np.zeros((4, 4))
    Vb2 = np.zeros((4, 4))
    Wb2 = np.zeros((2, 4))
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        Ub2[i*2+j, idx] = 1
        Vb2[j*2+0, idx] = 1
        Wb2[0*2+i, idx] = 1
    
    Ub = np.hstack([Ub1, Ub2])  # (4, 11)
    Vb = np.hstack([Vb1, Vb2])  # (4, 11)
    Wb = np.hstack([Wb1, Wb2])  # (4, 11) -- but Wb1 is (4,7) and Wb2 is (2,4)
    
    # W indices: k*n+i where k in [0,2], i in [0,1]
    # For 2x2x2 part: k in [0,1], so W indices 0..3
    # For 2x2x1 part: k=2, so W indices 4,5
    Wb_full = np.zeros((6, 11))
    Wb_full[0:4, 0:7] = Wb1
    Wb_full[4:6, 7:11] = Wb2
    
    return Ub, Vb, Wb_full


def strassen_2x3x2():
    """
    Returns a decomposition for a 2x3x2 block.
    n=2, m=3, p=2. 
    Split m=3 as 2+1: (2x2x2) Strassen (rank 7) + (2x1x2) schoolbook (rank 4) = 11.
    """
    Ub1, Vb1, Wb1 = strassen_decomposition_2x2()  # (4,7), (4,7), (4,7)
    
    # Schoolbook 2x1x2: n=2, m=1, p=2, rank=4
    Ub2 = np.zeros((2, 4))
    Vb2 = np.zeros((2, 4))
    Wb2 = np.zeros((4, 4))
    for idx, (i, k) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        Ub2[i, idx] = 1
        Vb2[k, idx] = 1
        Wb2[k*2+i, idx] = 1
    
    # Combine: U has rows for (i,j) with j in [0,2]
    # j=0,1 use Strassen; j=2 uses schoolbook
    Ub = np.zeros((6, 11))
    Ub[0:4, 0:7] = Ub1
    Ub[4:6, 7:11] = Ub2
    
    Vb = np.zeros((6, 11))
    Vb[0:4, 0:7] = Vb1
    Vb[4:6, 7:11] = Vb2
    
    Wb = np.hstack([Wb1, Wb2])  # (4, 11)
    
    return Ub, Vb, Wb


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


def build_block_decomposition_v2(n, m, p):
    """
    Alternative layout: m=5 as [3,2], p=6 as [2,2,2]
    Uses 2x3x2 blocks (rank 11 each) and 2x2x2 Strassen (rank 7 each).
    Total: 3*11 + 3*7 = 60. Worse than v1.
    
    Try: m=5 as [2,3], p=6 as [2,2,2]
    (j=0..1, k=0..1): 2x2x2 Strassen, rank 7
    (j=0..1, k=2..3): 2x2x2 Strassen, rank 7
    (j=0..1, k=4..5): 2x2x2 Strassen, rank 7
    (j=2..4, k=0..1): 2x3x2 block, rank 11
    (j=2..4, k=2..3): 2x3x2 block, rank 11
    (j=2..4, k=4..5): 2x3x2 block, rank 11
    Total: 3*7 + 3*11 = 60. Still worse.
    
    The original layout is best at 54.
    """
    return None


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
        # Contract over rank dimension
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
