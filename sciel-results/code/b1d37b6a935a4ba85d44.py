import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    deadline = time.time() + 118.0
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # We aim for rank 44: 4 blocks of Strassen (rank 7) + 4 blocks of schoolbook 1x2 (rank 4)
    # 4 * 7 + 4 * 4 = 28 + 16 = 44
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_rank7(a_row_off, b_col_off, j_off, c_row_off, c_col_off, col_start):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        A block rows [a_row_off, a_row_off+1], cols [j_off, j_off+1]
        B block rows [j_off, j_off+1], cols [b_col_off, b_col_off+1]
        C block rows [c_row_off, c_row_off+1], cols [c_col_off, c_col_off+1]
        """
        # Strassen terms for X(2x2) * Y(2x2) = Z(2x2)
        # Indices in local 2x2 block:
        # X: [[x00, x01], [x10, x11]] -> global U[(a_row_off+i)*m + (j_off+j)]
        # Y: [[y00, y01], [y10, y11]] -> global V[(j_off+i)*p + (b_col_off+j)]
        # Z: [[z00, z01], [z10, z11]] -> global W[(c_col_off+j)*n + (c_row_off+i)]
        
        # M1 = (x00+x11)(y00+y11) -> z00 +=, z11 +=
        c = col_start
        U[(a_row_off+0)*m + (j_off+0)][c] += 1
        U[(a_row_off+1)*m + (j_off+1)][c] += 1
        V[(j_off+0)*p + (b_col_off+0)][c] += 1
        V[(j_off+1)*p + (b_col_off+1)][c] += 1
        W[(c_col_off+0)*n + (c_row_off+0)][c] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][c] += 1
        col_start += 1
        
        # M2 = (x10+x11)y00 -> z10 +=, z11 -=
        c = col_start
        U[(a_row_off+1)*m + (j_off+0)][c] += 1
        U[(a_row_off+1)*m + (j_off+1)][c] += 1
        V[(j_off+0)*p + (b_col_off+0)][c] += 1
        W[(c_col_off+0)*n + (c_row_off+1)][c] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][c] -= 1
        col_start += 1
        
        # M3 = x00(y01-y11) -> z01 +=, z11 +=
        c = col_start
        U[(a_row_off+0)*m + (j_off+0)][c] += 1
        V[(j_off+0)*p + (b_col_off+1)][c] += 1
        V[(j_off+1)*p + (b_col_off+1)][c] -= 1
        W[(c_col_off+1)*n + (c_row_off+0)][c] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][c] += 1
        col_start += 1
        
        # M4 = x11(y10-y00) -> z00 +=, z10 +=
        c = col_start
        U[(a_row_off+1)*m + (j_off+1)][c] += 1
        V[(j_off+1)*p + (b_col_off+0)][c] += 1
        V[(j_off+0)*p + (b_col_off+0)][c] -= 1
        W[(c_col_off+0)*n + (c_row_off+0)][c] += 1
        W[(c_col_off+0)*n + (c_row_off+1)][c] += 1
        col_start += 1
        
        # M5 = (x00+x01)y11 -> z01 +=, z00 -=
        c = col_start
        U[(a_row_off+0)*m + (j_off+0)][c] += 1
        U[(a_row_off+0)*m + (j_off+1)][c] += 1
        V[(j_off+1)*p + (b_col_off+1)][c] += 1
        W[(c_col_off+1)*n + (c_row_off+0)][c] += 1
        W[(c_col_off+0)*n + (c_row_off+0)][c] -= 1
        col_start += 1
        
        # M6 = (x10-x00)(y00+y01) -> z11 +=
        c = col_start
        U[(a_row_off+1)*m + (j_off+0)][c] += 1
        U[(a_row_off+0)*m + (j_off+0)][c] -= 1
        V[(j_off+0)*p + (b_col_off+0)][c] += 1
        V[(j_off+0)*p + (b_col_off+1)][c] += 1
        W[(c_col_off+1)*n + (c_row_off+1)][c] += 1
        col_start += 1
        
        # M7 = (x01-x11)(y10+y11) -> z00 +=
        c = col_start
        U[(a_row_off+0)*m + (j_off+1)][c] += 1
        U[(a_row_off+1)*m + (j_off+1)][c] -= 1
        V[(j_off+1)*p + (b_col_off+0)][c] += 1
        V[(j_off+1)*p + (b_col_off+1)][c] += 1
        W[(c_col_off+0)*n + (c_row_off+0)][c] += 1
        col_start += 1
        
        return col_start

    def add_rank4(a_row, b_col_off, j_off, c_col_off, col_start):
        """
        Adds a rank-4 schoolbook decomposition for a 1x2 * 2x2 block.
        A block row [a_row], cols [j_off, j_off+1]
        B block rows [j_off, j_off+1], cols [b_col_off, b_col_off+1]
        C block row [a_row], cols [c_col_off, c_col_off+1]
        """
        # C[a_row, c_col_off] = A[a_row, j_off] * B[j_off, b_col_off] + A[a_row, j_off+1] * B[j_off+1, b_col_off]
        # C[a_row, c_col_off+1] = A[a_row, j_off] * B[j_off, b_col_off+1] + A[a_row, j_off+1] * B[j_off+1, b_col_off+1]
        
        # Term 1: A[a_row, j_off] * B[j_off, b_col_off] -> C[a_row, c_col_off]
        c = col_start
        U[a_row*m + (j_off+0)][c] = 1
        V[(j_off+0)*p + (b_col_off+0)][c] = 1
        W[(c_col_off+0)*n + a_row][c] = 1
        col_start += 1
        
        # Term 2: A[a_row, j_off+1] * B[j_off+1, b_col_off] -> C[a_row, c_col_off]
        c = col_start
        U[a_row*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (b_col_off+0)][c] = 1
        W[(c_col_off+0)*n + a_row][c] = 1
        col_start += 1
        
        # Term 3: A[a_row, j_off] * B[j_off, b_col_off+1] -> C[a_row, c_col_off+1]
        c = col_start
        U[a_row*m + (j_off+0)][c] = 1
        V[(j_off+0)*p + (b_col_off+1)][c] = 1
        W[(c_col_off+1)*n + a_row][c] = 1
        col_start += 1
        
        # Term 4: A[a_row, j_off+1] * B[j_off+1, b_col_off+1] -> C[a_row, c_col_off+1]
        c = col_start
        U[a_row*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (b_col_off+1)][c] = 1
        W[(c_col_off+1)*n + a_row][c] = 1
        col_start += 1
        
        return col_start

    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_rank7(0, 0, 0, 0, 0, col)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_rank7(0, 0, 2, 0, 0, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_rank7(0, 2, 0, 0, 2, col)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_rank7(0, 2, 2, 0, 2, col)
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    col = add_rank4(2, 0, 0, 0, col)
    
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    col = add_rank4(2, 0, 2, 0, col)
    
    # Block 7: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    col = add_rank4(2, 2, 0, 2, col)
    
    # Block 8: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4]
    col = add_rank4(2, 2, 2, 2, col)
    
    assert col == total_rank, f"Rank mismatch: {col} vs {total_rank}"

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        a_r = np.dot(U_arr[:, r], A_test.flatten())
        b_r = np.dot(V_arr[:, r], B_test.flatten())
        # W is indexed as W[k*n + i], so reshape to (p, n) and transpose to (n, p)
        C_computed += W_arr[:, r].reshape(p, n).T * (a_r * b_r)

    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
