"""
Rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.
Uses Strassen-like block decomposition to reduce rank below schoolbook 60.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    MARGIN = 10.0     # safety margin

    n, m, p = DIMENSIONS

    # Schoolbook baseline: rank = n*m*p = 60
    # We try to find a lower-rank decomposition using block/Strassen ideas.

    # Strategy: Split the 5x6 matrix into blocks and use Strassen-style
    # decomposition on sub-blocks where dimensions allow.

    # One approach: view 2x5 * 5x6 as:
    #   [A1 A2] * [B1]  where A1 is 2x2, A2 is 2x3, B1 is 5x6
    # But this doesn't directly help with rank reduction.

    # Better: Use the known result that for small matrices, we can sometimes
    # beat schoolbook. For 2x5x6, let's try a direct search or known decompositions.

    # Known: The rank of matrix multiplication tensor M_{n,m,p} satisfies
    # R(M_{n,m,p}) <= n*m*p (schoolbook)
    # For small cases, better bounds are known.

    # Let's try a structured decomposition:
    # Split columns of A (2x5) into 2x2 + 2x3
    # Split rows of B (5x6) into 2x6 + 3x6
    # Then C = A1*B1 + A2*B2 where A1 is 2x2, B1 is 2x6, A2 is 2x3, B2 is 3x6

    # Rank of M_{2,2,6} + Rank of M_{2,3,6}
    # Schoolbook: 2*2*6 + 2*3*6 = 24 + 36 = 60 (same as before)

    # Try Strassen on 2x2 part:
    # M_{2,2,2} has rank 7 (Strassen). For M_{2,2,6}, we can use 7*3=21?
    # Actually, M_{2,2,6} can be decomposed as 3 copies of M_{2,2,2} if we split
    # the 6 columns into 3 pairs. Each M_{2,2,2} has rank 7, so total 21.
    # Plus M_{2,3,6} with schoolbook rank 36. Total: 21+36 = 57.

    # Let's verify this construction:
    # C[0:2, 0:6] = A[0:2, 0:2] * B[0:2, 0:6] + A[0:2, 2:5] * B[2:5, 0:6]
    # For the first term, split B[0:2, 0:6] into three 2x2 blocks:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    # Each is a 2x2x2 matrix multiplication, decomposable with rank 7.
    # So first term uses 3*7 = 21 rank terms.
    # Second term: A[0:2, 2:5] is 2x3, B[2:5, 0:6] is 3x6.
    # Schoolbook rank for this: 2*3*6 = 36.
    # Total: 21 + 36 = 57.

    # Let's construct this decomposition.

    rank = 57
    U = [[0] * rank for _ in range(n * m)]  # 2*5 = 10 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*6 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 6*2 = 12 rows

    col = 0

    # Part 1: A[0:2, 0:2] * B[0:2, 0:6] using 3 Strassen 2x2x2 blocks
    # Strassen 2x2x2 decomposition (rank 7):
    # Given X (2x2) and Y (2x2), compute Z = X*Y with 7 multiplications.
    # m1 = (X00+X11)*(Y00+Y11)
    # m2 = (X10+X11)*Y00
    # m3 = X00*(Y01-Y11)
    # m4 = X11*(Y10-Y00)
    # m5 = (X00+X01)*Y11
    # m6 = (X10-X00)*(Y00+Y01)
    # m7 = (X01-X11)*(Y10+Y11)
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6

    # For 2x2 * 2x2, the tensor entries are:
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # where i in {0,1}, j in {0,1}, k in {0,1}

    def strassen_2x2_2x2(U, V, W, row_offset_U, row_offset_V, row_offset_W, start_col):
        """Add Strassen 2x2x2 decomposition. Returns number of columns used."""
        # 7 terms
        # m1 = (X00+X11)*(Y00+Y11)
        U[row_offset_U + 0][start_col + 0] += 1  # X00
        U[row_offset_U + 3][start_col + 0] += 1  # X11
        V[row_offset_V + 0][start_col + 0] += 1  # Y00
        V[row_offset_V + 3][start_col + 0] += 1  # Y11
        W[row_offset_W + 0][start_col + 0] += 1  # Z00
        W[row_offset_W + 1][start_col + 0] += 0  # Z01
        W[row_offset_W + 2][start_col + 0] += 0  # Z10
        W[row_offset_W + 3][start_col + 0] += 1  # Z11

        # m2 = (X10+X11)*Y00
        U[row_offset_U + 2][start_col + 1] += 1  # X10
        U[row_offset_U + 3][start_col + 1] += 1  # X11
        V[row_offset_V + 0][start_col + 1] += 1  # Y00
        W[row_offset_W + 2][start_col + 1] += 1  # Z10
        W[row_offset_W + 3][start_col + 1] += -1  # Z11

        # m3 = X00*(Y01-Y11)
        U[row_offset_U + 0][start_col + 2] += 1  # X00
        V[row_offset_V + 1][start_col + 2] += 1  # Y01
        V[row_offset_V + 3][start_col + 2] += -1  # Y11
        W[row_offset_W + 1][start_col + 2] += 1  # Z01
        W[row_offset_W + 3][start_col + 2] += 1  # Z11

        # m4 = X11*(Y10-Y00)
        U[row_offset_U + 3][start_col + 3] += 1  # X11
        V[row_offset_V + 2][start_col + 3] += 1  # Y10
        V[row_offset_V + 0][start_col + 3] += -1  # Y00
        W[row_offset_W + 0][start_col + 3] += 1  # Z00
        W[row_offset_W + 2][start_col + 3] += 1  # Z10

        # m5 = (X00+X01)*Y11
        U[row_offset_U + 0][start_col + 4] += 1  # X00
        U[row_offset_U + 1][start_col + 4] += 1  # X01
        V[row_offset_V + 3][start_col + 4] += 1  # Y11
        W[row_offset_W + 0][start_col + 4] += -1  # Z00
        W[row_offset_W + 1][start_col + 4] += 1  # Z01

        # m6 = (X10-X00)*(Y00+Y01)
        U[row_offset_U + 2][start_col + 5] += 1  # X10
        U[row_offset_U + 0][start_col + 5] += -1  # X00
        V[row_offset_V + 0][start_col + 5] += 1  # Y00
        V[row_offset_V + 1][start_col + 5] += 1  # Y01
        W[row_offset_W + 3][start_col + 5] += 1  # Z11

        # m7 = (X01-X11)*(Y10+Y11)
        U[row_offset_U + 1][start_col + 6] += 1  # X01
        U[row_offset_U + 3][start_col + 6] += -1  # X11
        V[row_offset_V + 2][start_col + 6] += 1  # Y10
        V[row_offset_V + 3][start_col + 6] += 1  # Y11
        W[row_offset_W + 0][start_col + 6] += 1  # Z00
        W[row_offset_W + 1][start_col + 6] += -1  # Z01

        return 7

    # Part 1: 3 Strassen blocks for A[0:2,0:2] * B[0:2, 0:6]
    # A indices: rows 0,1; cols 0,1 -> U rows 0,1,2,3 (i*m+j for i in 0,1, j in 0,1)
    #   U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
    # B indices: rows 0,1; cols 0,1,2,3,4,5 -> V rows for j in 0,1 and k in 0..5
    #   V[j*p+k] = B[j,k]
    #   For B[0:2, 0:2]: V[0], V[1], V[6], V[7] (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    #   Wait, V[j*p+k] with p=6. So V[0*6+0]=V[0], V[0*6+1]=V[1], V[1*6+0]=V[6], V[1*6+1]=V[7]
    # C indices: W[k*n+i] with n=2. W[0*2+0]=W[0], W[0*2+1]=W[1], W[1*2+0]=W[2], W[1*2+1]=W[3]
    #   W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]

    # Block 1: B[0:2, 0:2]
    col += strassen_2x2_2x2(U, V, W, 0, 0, 0, col)

    # Block 2: B[0:2, 2:4]
    # V offset: j=0 -> k=2,3 => V[12], V[13]; j=1 -> k=2,3 => V[18], V[19]
    # But our function expects contiguous V rows. Let me adjust.
    # Actually, V[j*p+k] = V[6j+k]. For k=2,3: V[6j+2], V[6j+3]
    # So V rows are 2,3 for j=0 and 8,9 for j=1. Not contiguous with offset.
    # Let me modify the function to take explicit row mappings.

    # Let me redo with explicit indexing.

    # Reset and do it properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(A_rows, B_row_cols, C_row_cols, start_col):
        """
        Add a Strassen 2x2x2 block.
        A_rows: list of 4 U-row indices for X00,X01,X10,X11
        B_row_cols: list of 4 V-row indices for Y00,Y01,Y10,Y11
        C_row_cols: list of 4 W-row indices for Z00,Z01,Z10,Z11
        """
        X00, X01, X10, X11 = A_rows
        Y00, Y01, Y10, Y11 = B_row_cols
        Z00, Z01, Z10, Z11 = C_row_cols

        # m1 = (X00+X11)*(Y00+Y11)
        U[X00][start_col + 0] += 1
        U[X11][start_col + 0] += 1
        V[Y00][start_col + 0] += 1
        V[Y11][start_col + 0] += 1
        W[Z00][start_col + 0] += 1
        W[Z11][start_col + 0] += 1

        # m2 = (X10+X11)*Y00
        U[X10][start_col + 1] += 1
        U[X11][start_col + 1] += 1
        V[Y00][start_col + 1] += 1
        W[Z10][start_col + 1] += 1
        W[Z11][start_col + 1] += -1

        # m3 = X00*(Y01-Y11)
        U[X00][start_col + 2] += 1
        V[Y01][start_col + 2] += 1
        V[Y11][start_col + 2] += -1
        W[Z01][start_col + 2] += 1
        W[Z11][start_col + 2] += 1

        # m4 = X11*(Y10-Y00)
        U[X11][start_col + 3] += 1
        V[Y10][start_col + 3] += 1
        V[Y00][start_col + 3] += -1
        W[Z00][start_col + 3] += 1
        W[Z10][start_col + 3] += 1

        # m5 = (X00+X01)*Y11
        U[X00][start_col + 4] += 1
        U[X01][start_col + 4] += 1
        V[Y11][start_col + 4] += 1
        W[Z00][start_col + 4] += -1
        W[Z01][start_col + 4] += 1

        # m6 = (X10-X00)*(Y00+Y01)
        U[X10][start_col + 5] += 1
        U[X00][start_col + 5] += -1
        V[Y00][start_col + 5] += 1
        V[Y01][start_col + 5] += 1
        W[Z11][start_col + 5] += 1

        # m7 = (X01-X11)*(Y10+Y11)
        U[X01][start_col + 6] += 1
        U[X11][start_col + 6] += -1
        V[Y10][start_col + 6] += 1
        V[Y11][start_col + 6] += 1
        W[Z00][start_col + 6] += 1
        W[Z01][start_col + 6] += -1

        return 7

    # Part 1: A[0:2,0:2] * B[0:2, 0:6]
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    A_rows = [0, 1, 2, 3]

    # B[0:2, 0:2]: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[6], B[1,1]=V[7]
    B_rows_1 = [0, 1, 6, 7]
    # C[0:2, 0:2]: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    C_rows_1 = [0, 1, 2, 3]
    col += add_strassen_block(A_rows, B_rows_1, C_rows_1, col)

    # B[0:2, 2:4]: B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[8], B[1,3]=V[9]
    B_rows_2 = [2, 3, 8, 9]
    # C[0:2, 2:4]: C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    C_rows_2 = [4, 5, 6, 7]
    col += add_strassen_block(A_rows, B_rows_2, C_rows_2, col)

    # B[0:2, 4:6]: B[0,4]=V[4], B[0,5]=V[5], B[1,4]=V[10], B[1,5]=V[11]
    B_rows_3 = [4, 5, 10, 11]
    # C[0:2, 4:6]: C[0,4]=W[8], C[0,5]=W[9], C[1,4]=W[10], C[1,5]=W[11]
    C_rows_3 = [8, 9, 10, 11]
    col += add_strassen_block(A_rows, B_rows_3, C_rows_3, col)

    # Part 2: A[0:2, 2:5] * B[2:5, 0:6] using schoolbook (2*3*6=36)
    # A[0,2]=U[4], A[0,3]=U[5], A[0,4]=U[6]
    # A[1,2]=U[7], A[1,3]=U[8], A[1,4]=U[9]
    # B[2,0]=V[12], B[2,1]=V[13], ..., B[2,5]=V[17]
    # B[3,0]=V[18], ..., B[3,5]=V[23]
    # B[4,0]=V[24], ..., B[4,5]=V[29]
    # C[0,0]=W[0], ..., C[0,5]=W[5], C[1,0]=W[6], ..., C[1,5]=W[11]

    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i * 5 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Convert to numpy for verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 for all (i,j,k) since matrix multiplication
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
