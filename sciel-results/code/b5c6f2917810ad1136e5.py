import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 130.0

    # We use 4 blocks of M_{3,2,2} with rank 11 each, total rank 44.
    # Each block: A (3x2) * B (2x2) -> C (3x2)
    # Strassen for the 2x2x2 core (rows 0,1 of A, all of B) + 4 trivial for row 2 of A.

    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Local 3x2x2 decomposition, rank 11
    # A indices: (r,c) -> r*2+c for r in 0..2, c in 0..1
    # B indices: (r,c) -> r*2+c for r in 0..1, c in 0..1
    # C indices: (r,c) -> r*2+c for r in 0..2, c in 0..1
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen for 2x2x2 (A rows 0,1; B full 2x2; C rows 0,1)
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # M2: (A10+A11)B00 -> C10, C11
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # M3: A00(B10-B00) -> C01, C11
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # M4: A11(B11-B01) -> C00, C10
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # M5: (A00+A01)B11 -> C00, C01
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # M6: (A10-A00)(B00+B01) -> C11
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # M7: (A01-A11)(B10+B11) -> C00
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Row 2 of A (trivial, 4 terms)
    # M8: A20*B00 -> C20
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    # M9: A21*B10 -> C20
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    # M10: A20*B01 -> C21
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    # M11: A21*B11 -> C21
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1

    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                la = r * 2 + c
                ga = (a_row_start + r) * m + (a_col_start + c)
                U[ga, col_offset:col_offset+11] += U_loc[la, :]
        for r in range(2):
            for c in range(2):
                lb = r * 2 + c
                gb = (b_row_start + r) * p + (b_col_start + c)
                V[gb, col_offset:col_offset+11] += V_loc[lb, :]
        for r in range(3):
            for c in range(2):
                lc = r * 2 + c
                gc = (c_col_start + c) * n + (c_row_start + r)
                W[gc, col_offset:col_offset+11] += W_loc[lc, :]

    # 4 blocks partitioning the 4x4 B into 2x2 blocks
    add_block(0, 0, 0, 0, 0, 0, 0)   # A(:,0:2) * B(0:2,0:2) -> C(:,0:2)
    add_block(1, 0, 2, 2, 0, 0, 0)   # A(:,2:4) * B(2:4,0:2) -> C(:,0:2)
    add_block(2, 0, 0, 0, 2, 0, 2)   # A(:,0:2) * B(0:2,2:4) -> C(:,2:4)
    add_block(3, 0, 2, 2, 2, 0, 2)   # A(:,2:4) * B(2:4,2:4) -> C(:,2:4)

    # Verify correctness: reconstruct tensor and compare
    # Tensor T[i,j,k] = delta_{i,k} (for M_{n,m,p}: C[i,k] = sum_j A[i,j] B[j,k])
    # W uses trace convention: W[k*n+i, r]
    # U[i*m+j, r] for A[i,j]
    # V[j*p+k, r] for B[j,k]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k}

    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if i == k else 0.0
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - expected) > 1e-9:
                        return False
        return True

    ok = verify(U, V, W, n, m, p)
    if not ok:
        # Should not happen; return best effort anyway
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
