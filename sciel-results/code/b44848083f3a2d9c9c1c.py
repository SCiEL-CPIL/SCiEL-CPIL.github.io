"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)

    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False

    # Target tensor: T[a*m+b, b*p+c, c*n+a] = 1 for all (a,b,c)
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1

    # T_computed[i,j,k] = sum_c U2[i,c] * V2[j,c] * W2[k,c]
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)

    # Each factor scaled by 2, product scaled by 8
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True


def _build_strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: A=(A00+A11), B=(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2: A=(A10+A11), B=B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3: A=A00, B=(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4: A=A11, B=(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5: A=(A00+A01), B=B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6: A=(A10-A00), B=(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7: A=(A01-A11), B=(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's 7-term formula applied at the 2x2 block level.
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _build_strassen_2x2()

    # Strassen macro terms: for each of the 7 macro products,
    # list of (block_index, coefficient) for the A-side and B-side,
    # and the C-side block contributions.
    # block_index: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    # A-side uses (row_block, col_block); B-side uses (row_block, col_block).
    macro_terms = [
        # (A_contribs, B_contribs, C_contribs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    def a_global_index(block_row, block_col, local_row, local_col):
        """Global index for A element at (block_row, block_col) + (local_row, local_col)."""
        return (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)

    def b_global_index(block_row, block_col, local_row, local_col):
        """Global index for B element at (block_row, block_col) + (local_row, local_col)."""
        return (block_row * 2 + local_row) * 4 + (block_col * 2 + local_col)

    def c_global_index(block_col, block_row, local_col, local_row):
        """Global index for C element at (block_col, block_row) + (local_col, local_row).
        C is indexed as C[row, col] where row=block_row*2+local_row, col=block_col*2+local_col.
        But in our convention, C index = c*n + a where c is column, a is row.
        So C global = col * 4 + row = (block_col*2+local_col)*4 + (block_row*2+local_row).
        """
        return (block_col * 2 + local_col) * 4 + (block_row * 2 + local_row)

    col = 0
    for macro_k in range(7):
        a_contribs = macro_terms[macro_k][0]
        b_contribs = macro_terms[macro_k][1]
        c_contribs = macro_terms[macro_k][2]

        for micro_l in range(7):
            for local_idx in range(4):
                u_coeff = U2[local_idx][micro_l]
                v_coeff = V2[local_idx][micro_l]
                w_coeff = W2[local_idx][micro_l]

                local_row, local_col = local_idx // 2, local_idx % 2

                # U contributions
                for (a_blk_r, a_blk_c) in a_contribs:
                    if u_coeff != 0:
                        global_idx = a_global_index(a_blk_r, a_blk_c, local_row, local_col)
                        U[global_idx][col] += u_coeff

                # V contributions
                for (b_blk_r, b_blk_c) in b_contribs:
                    if v_coeff != 0:
                        global_idx = b_global_index(b_blk_r, b_blk_c, local_row, local_col)
                        V[global_idx][col] += v_coeff

                # W contributions
                for (c_blk_r, c_blk_c) in c_contribs:
                    if w_coeff != 0:
                        global_idx = c_global_index(c_blk_c, c_blk_r, local_col, local_row)
                        W[global_idx][col] += w_coeff

            col += 1

    assert col == 49, f"Expected 49 columns, got {col}"
    return U, V, W


def _build_5x5_decomposition():
    """
    Build a rank-110 decomposition for 5x5x5 matrix multiplication.
    Strategy: Split 5 = 4 + 1.
    - Use rank-49 Strassen for the 4x4 sub-block.
    - Handle cross terms between 4-part and 1-part naively.
    """
    U4, V4, W4 = _build_strassen_4x4()

    rank = 49 + 61  # 110
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]

    # First 49 columns: 4x4 Strassen
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]

    # Remaining 61 columns: cross terms (naive)
    col = 49
    for i in range(5):
        for k in range(5):
            for j in range(5):
                if i < 4 and k < 4 and j < 4:
                    continue
                U[i*5 + k][col] = 1
                V[k*5 + j][col] = 1
                W[j*5 + i][col] = 1
                col += 1

    assert col == 110, f"Expected 110 columns, got {col}"
    return U, V, W


def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS

    # Build the rank-110 decomposition
    U, V, W = _build_5x5_decomposition()

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 125
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
