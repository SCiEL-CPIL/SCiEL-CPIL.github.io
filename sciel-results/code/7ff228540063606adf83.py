"""Optimized rank-28 Strassen decomposition for 2x4 x 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 130  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        """Standard Strassen decomposition for 2x2x2 matrix multiplication.
        U, V, W are 4x7 matrices where the 4 entries are indexed as (i*2+j)
        for the 2x2 blocks, and 7 is the rank.
        """
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # C[0,0] -> index 0, C[0,1] -> index 1, C[1,0] -> index 2, C[1,1] -> index 3
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        # M2 = (A10+A11)B00
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        # M3 = A00(B10-B11)
        U[0,2]=1.0; V[2,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        # M4 = A11(B01-B00)
        U[3,3]=1.0; V[1,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        # M5 = (A00+A01)B11
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        """Build rank-28 decomposition using 4 independent Strassen 2x2 blocks."""
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))

        # 2x4 x 4x4 can be split into 4 blocks of 2x2 x 2x2
        # A is 2x4, B is 4x4, C is 2x4
        # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2]
        # C[:, 0:2] += A[:, 2:4] @ B[2:4, 0:2]
        # C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4]
        # C[:, 2:4] += A[:, 2:4] @ B[2:4, 2:4]

        blocks = [
            # (A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols)
            (0, 2, 0, 2, 0, 2),
            (0, 2, 2, 4, 0, 2),
            (2, 4, 0, 2, 0, 2),
            (2, 4, 2, 4, 0, 2),
        ]

        for bi, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
            for t in range(7):
                col = bi * 7 + t
                for i in range(2):
                    for j in range(2):
                        # A index: i (row in 2x2 block), j (col in 2x2 block)
                        # Global A index: ar+i, ac+j
                        # Global B index: br+j, bc+k
                        # Global C index: cr+i, cc+k
                        U_full[ar + i, col] = U_s[i * 2 + j, t]
                        V_full[br + j, col] = V_s[j * 2 + 0, t]  # placeholder
                        V_full[br + j, col] = V_s[j * 2 + 0, t]
                # Reset and do it properly
            # Let me redo this more carefully
            for t in range(7):
                col = bi * 7 + t
                # U: A block indices (i, j) -> global (ar+i, ac+j), Strassen index i*2+j
                for i in range(2):
                    for j in range(2):
                        U_full[ar + i, col] = U_s[i * 2 + j, t]
                # V: B block indices (j, k) -> global (br+j, bc+k), Strassen index j*2+k
                for j in range(2):
                    for k in range(2):
                        V_full[br + j, col] = V_s[j * 2 + k, t]
                # W: C block indices (k, i) -> global (cr+i, cc+k), Strassen index i*2+k
                # Wait, C is 2x4, so C[cr+i, cc+k] where i in 0..1, k in 0..1
                # The Strassen W index is for the 2x2 output, indexed as (row, col) -> row*2+col
                # But in our tensor, C is indexed as (k*n+i) where n=2, so C_index = k*2+i
                # For the 2x2 block C[cr+i, cc+k], the global index is cr+i for row, cc+k for col
                # But the tensor index for C is just a single index: c = k*n + i = k*2 + i
                # Wait, this is getting confusing. Let me re-examine.
                pass

        # Let me rebuild this more carefully from scratch
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))

        # The matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # where i in 0..1, j in 0..3, k in 0..3
        # Tensor index: a = i*m+j = i*4+j, b = j*p+k = j*4+k, c = k*n+i = k*2+i

        # Split into 4 blocks:
        # Block 0: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
        # Block 1: A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
        # Block 2: A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
        # Block 3: A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
        # Wait, that's only A[0:2, :]. I need A[1:2, :] too.
        # Actually, A is 2x4, so A[0, :] and A[1, :] are the two rows.
        # Let me think of it as: A = [A_00, A_01] where A_00 is 2x2 (rows 0:2, cols 0:2) and A_01 is 2x2 (rows 0:2, cols 2:4)
        # B = [[B_00, B_01], [B_10, B_11]] where each is 2x2
        # C = A @ B = [A_00 @ B_00 + A_01 @ B_10, A_00 @ B_01 + A_01 @ B_11]
        # C is 2x4, so C[:, 0:2] = A_00 @ B_00 + A_01 @ B_10, C[:, 2:4] = A_00 @ B_01 + A_01 @ B_11

        # So 4 Strassen decompositions:
        # 1. A_00 @ B_00 -> contributes to C[:, 0:2]
        # 2. A_00 @ B_01 -> contributes to C[:, 2:4]
        # 3. A_01 @ B_10 -> contributes to C[:, 0:2]
        # 4. A_01 @ B_11 -> contributes to C[:, 2:4]

        blocks = [
            # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
            (0, 0, 0, 0, 0, 0),   # A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
            (0, 0, 0, 2, 0, 2),   # A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
            (0, 2, 2, 0, 0, 0),   # A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
            (0, 2, 2, 2, 0, 2),   # A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
        ]

        for bi, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(blocks):
            for t in range(7):
                col = bi * 7 + t
                for i in range(2):
                    for j in range(2):
                        # A[i, j] in the 2x2 block -> global A[a_r+i, a_c+j]
                        # Tensor index a = (a_r+i)*m + (a_c+j) = (a_r+i)*4 + (a_c+j)
                        U_full[(a_r + i) * 4 + (a_c + j), col] = U_s[i * 2 + j, t]
                for j in range(2):
                    for k in range(2):
                        # B[j, k] in the 2x2 block -> global B[b_r+j, b_c+k]
                        # Tensor index b = (b_r+j)*p + (b_c+k) = (b_r+j)*4 + (b_c+k)
                        V_full[(b_r + j) * 4 + (b_c + k), col] = V_s[j * 2 + k, t]
                for i in range(2):
                    for k in range(2):
                        # C[i, k] in the 2x2 block -> global C[c_r+i, c_c+k]
                        # Tensor index c = (c_c+k)*n + (c_r+i) = (c_c+k)*2 + (c_r+i)
                        W_full[(c_c + k) * 2 + (c_r + i), col] = W_s[i * 2 + k, t]

        return U_full, V_full, W_full

    def build_schoolbook():
        """Fallback: schoolbook rank-32 decomposition."""
        rank = n * m * p  # 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U28, V28, W28 = build_strassen28()

    if verify(U28, V28, W28, tol=1e-8):
        return to_lists(U28, V28, W28, 28)

    # Fallback to schoolbook if Strassen verification fails
    U_sb, V_sb, W_sb = build_schoolbook()
    return to_lists(U_sb, V_sb, W_sb, 32)
# EVOLVE-BLOCK-END
