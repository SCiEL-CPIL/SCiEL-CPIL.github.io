import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=float)  # shape (n*m, rank)
    V_mat = np.array(V, dtype=float)  # shape (m*p, rank)
    W_mat = np.array(W, dtype=float)  # shape (p*n, rank)

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _build_rank20():
    """
    Build a rank-20 decomposition for 2x4 * 4x8 -> 2x8 by embedding into
    4x4 * 4x4 matrix multiplication (Strassen-based rank-20) and zero-padding.
    """
    n, m, p = 2, 4, 8
    N = 4  # embedded size

    # Zero-pad A (2x4) -> A4 (4x4): A4[i,j] = A[i,j] if i<2, j<4 else 0
    # Zero-pad B (4x8) -> B4 (4x4): B4[j,k] = B[j,k] if k<4 else 0 (k = 0..3)
    # Output C4 (4x4): C4[i,k] = C[i,k] if i<2, k<4 else 0
    # So we only need C[i,k] for i in {0,1}, k in {0,1,2,3}

    # Strassen 2x2 factors (rank 7)
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)  # (4, 7)

    V2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)  # (4, 7)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)  # (4, 7)

    # 4x4 Strassen: rank 20
    # Blocks of 4x4: A4 = [[A00,A01],[A10,A11]] each 2x2
    # B4 = [[B00,B01],[B10,B11]] each 2x2
    # C4 = A4 @ B4

    # 4x4 Strassen intermediate matrices (rank 20):
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11(B10-B00)
    # M3 = (A00-A10)A01
    # M4 = (A11-A11)B11  -- actually (A11-A11) doesn't make sense; let me use standard
    # Standard 4x4 Strassen (rank 20):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # M8..M20: each Mi is a 2x2 Strassen product (7 rank)
    # Actually 4x4 Strassen uses 7 sub-products, each of rank 7 = 49? No.
    # Standard 4x4 Strassen: 7 multiplications of 2x2 matrices, each 2x2 mult has rank 7
    # Total rank = 7 * 7 = 49? That's not 20.
    #
    # Wait: rank of 4x4 matrix multiplication. The known best is 49 (Strassen recursion)
    # or 48 (Lader). NOT 20. I was confused.
    #
    # Let me reconsider. The task says "rank-20 decomposition for 4x4 matrix multiplication
    # (derived from Strassen recursion)". That's not standard. Let me just use the
    # block-Strassen approach more carefully.

    # Actually, let me use a different approach: direct tensor decomposition search.
    # For 2x4 x 4x8, the tensor has 2*4*8 = 64 entries.
    # Let me try to find a low-rank decomposition via iterative optimization.

    return None


def _build_block_strassen():
    """
    Build the rank-56 block-Strassen decomposition (known to work).
    2x4 * 4x8: split into 2 blocks of 2x2 * 2x8, each 2x8 = 4 blocks of 2x2.
    Each 2x2*2x2 uses Strassen rank 7.
    Total: 2 * 4 * 7 = 56.
    """
    n, m, p = 2, 4, 8
    rank = 56

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ]

    V_strassen = [
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # Split A into two 2x2 blocks: A_left = A[:,0:2], A_right = A[:,2:4]
    # B is 4x8: rows j=0..3, cols k=0..7
    # Split B into top 2x8 and bottom 2x8: B_top = B[0:2,:], B_bot = B[2:4,:]
    # C is 2x8: C[i,k] = sum_j A[i,j]*B[j,k]
    #   = sum_{j=0}^1 A[i,j]*B[j,k] + sum_{j=2}^3 A[i,j]*B[j,k]
    #   = (A_left @ B_top)[i,k] + (A_right @ B_bot)[i,k]

    # Each 2x2 * 2x8 = 4 blocks of 2x2 * 2x2
    # For A_left (2x2) * B_top (2x8): 4 sub-blocks
    #   B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Each sub-product is 2x2, uses Strassen rank 7

    col_offset = 0
    for block in range(4):  # 4 sub-blocks for left part
        for f in range(7):
            col = col_offset + f
            # U: A_left[i,j] where i,j in {0,1} -> index i*2+j in 0..3
            for r in range(4):
                U[r][col] = U_strassen[f][r]
            # V: B_top[j, k] where j in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_strassen[f][r]
            # W: C[i, k] where i in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_strassen[r][f]
        col_offset += 7

    for block in range(4):  # 4 sub-blocks for right part
        for f in range(7):
            col = col_offset + f
            # U: A_right[i,j] where i,j in {0,1} -> index i*2+2+j in 4..7
            for r in range(4):
                U[2 + r][col] = U_strassen[f][r]
            # V: B_bot[j, k] where j in {2,3}, k = 2*block..2*block+1
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_strassen[f][r]
            # W: C[i, k] where i in {0,1}, k = 2*block..2*block+1
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_strassen[r][f]
        col_offset += 7

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return U_list, V_list, W_list


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Try to find a lower-rank decomposition via optimization
    # First, get the known rank-56 solution
    U, V, W = _build_block_strassen()

    if not _verify_fast(U, V, W, n, m, p, trials=30, tol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to reduce rank via column elimination
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Greedy column elimination: try removing each column and check validity
    for trial in range(30):
        if time.time() - start_time > deadline - 10:
            break

        # Try random subsets of columns of decreasing size
        for target_rank in range(best_rank - 1, max(10, best_rank - 10), -1):
            if time.time() - start_time > deadline - 10:
                break
            for attempt in range(5):
                cols = np.random.choice(range(best_rank), size=target_rank, replace=False)
                U_test = U_arr[:, cols]
                V_test = V_arr[:, cols]
                W_test = W_arr[:, cols]

                if _verify_fast(U_test.tolist(), V_test.tolist(), W_test.tolist(),
                                n, m, p, trials=15, tol=1e-9):
                    best_U = U_test.tolist()
                    best_V = V_test.tolist()
                    best_W = W_test.tolist()
                    best_rank = target_rank
                    break
            if time.time() - start_time > deadline - 10:
                break

    if time.time() - start_time > deadline - 5:
        return {"U": U, "V": V, "W": W}

    return {"U": best_U, "V": best_V, "W": best_W}
