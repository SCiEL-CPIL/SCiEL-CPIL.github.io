"""
Optimized Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
Achieves Rank 28.
"""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Target Tensor Definition
    # T[a,b,c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Strassen Rank-7 Decomposition for 2x2 * 2x2
    # Indices for 2x2 block:
    # A: 0,1,2,3 (i=0,1; j=0,1) -> i*2+j
    # B: 0,1,2,3 (j=0,1; k=0,1) -> j*2+k
    # C: 0,1,2,3 (k=0,1; i=0,1) -> k*2+i (Trace convention: c = k*n + i, n=2)
    # C[k,i] index = k*2 + i
    # C00 (k=0,i=0) -> 0
    # C10 (k=0,i=1) -> 1
    # C01 (k=1,i=0) -> 2
    # C11 (k=1,i=1) -> 3

    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))

    # M1 = (A00+A11)*(B00+B11) -> C00, C11
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1; W_22[3, 0] = 1

    # M2 = (A10+A11)*B00 -> C10, C11
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[1, 1] = 1; W_22[3, 1] = -1

    # M3 = A00*(B01-B11) -> C01, C11
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[2, 2] = 1; W_22[3, 2] = 1

    # M4 = A11*(B10-B00) -> C00, C10
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    W_22[0, 3] = 1; W_22[1, 3] = 1

    # M5 = (A00+A01)*B11 -> C00, C01
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[0, 4] = -1; W_22[2, 4] = 1

    # M6 = (A10-A00)*(B00+B01) -> C11
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    W_22[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11) -> C00
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1

    # Construct Full Rank 28 (4 blocks of 7)
    full_rank = 28
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))

    # Block (li, lk) where li in {0,1} (row block of A), lk in {0,1} (col block of B)
    # A row block li corresponds to i=li? No.
    # A is 2x4. Rows i=0,1. Cols j=0,1,2,3.
    # B is 4x4. Rows j=0,1,2,3. Cols k=0,1,2,3.
    # C is 2x4. Rows i=0,1. Cols k=0,1,2,3.
    
    # We split B into 4 2x2 blocks based on j and k.
    # Block 1: j in {0,1}, k in {0,1}. Multiplied by A[:, 0:2]. Result C[:, 0:2].
    # Block 2: j in {0,1}, k in {2,3}. Multiplied by A[:, 0:2]. Result C[:, 2:4].
    # Block 3: j in {2,3}, k in {0,1}. Multiplied by A[:, 2:4]. Result C[:, 0:2].
    # Block 4: j in {2,3}, k in {2,3}. Multiplied by A[:, 2:4]. Result C[:, 2:4].
    
    # Note on Strassen 2x2 * 2x2 mapping:
    # Local A (2x2): rows i_loc, cols j_loc.
    # Local B (2x2): rows j_loc, cols k_loc.
    # Local C (2x2): rows i_loc, cols k_loc.
    # Global indices:
    # A_global row = i_global (0 or 1).
    # A_global col = i_loc * 4 + j_global.
    # Wait, A is 2x4. Row index is i. Col index is j.
    # In block calculation, we take A[i, j].
    # For a 2x2 block of A (cols j0..j1), and B (rows j0..j1, cols k0..k1),
    # the multiplication is (A_sub) * (B_sub) = C_sub.
    # A_sub is 2x2. B_sub is 2x2. C_sub is 2x2.
    
    blocks = [
        (0, 0), # j=0..1, k=0..1 -> C cols 0..1
        (1, 0), # j=2..3, k=0..1 -> C cols 0..1 (Wait, B rows j=2..3, A cols j=2..3)
        (0, 1), # j=0..1, k=2..3 -> C cols 2..3
        (1, 1), # j=2..3, k=2..3 -> C cols 2..3
    ]
    
    # Let's trace carefully:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We can split sum_j into sum_{j in 0..1} + sum_{j in 2..3}.
    # Term 1: sum_{j=0,1} A_{i,j} B_{j,k}. This is (A[:, 0:2]) * (B[0:2, k]).
    #   If we fix k, this is dot product of A[:, 0:2] and B[0:2, k].
    #   Using 2x2 Strassen, we split B[0:2, :] into 2x2 blocks of k.
    #   So for j_block=0 (j=0,1), we have two k_blocks: k=0,1 and k=2,3.
    #   This uses 2 * Rank(2x2*2x2) = 14 terms.
    # Term 2: sum_{j=2,3} A_{i,j} B_{j,k}.
    #   This uses A[:, 2:4] and B[2:4, :].
    #   Again two k_blocks. 14 terms.
    # Total 28 terms.

    for b_idx, (j_off, k_off) in enumerate([
        (0, 0), (0, 2), # From A[:, 0:2] * B[0:2, :]
        (2, 0), (2, 2)  # From A[:, 2:4] * B[2:4, :]
    ]):
        # j_off: start row of B / start col of A
        # k_off: start col of B / start col of C
        col_start = b_idx * 7
        for t in range(7):
            col = col_start + t
            # A mapping: local row i_loc (0,1) -> global i (0,1). local col j_loc (0,1) -> global col j_off + j_loc.
            for i_loc in range(2):
                for j_loc in range(2):
                    local_a_idx = i_loc * 2 + j_loc
                    global_a_idx = i_loc * 4 + (j_off + j_loc)
                    U_full[global_a_idx, col] = U_22[local_a_idx, t]
            
            # B mapping: local row j_loc (0,1) -> global row j_off + j_loc. local col k_loc (0,1) -> global col k_off + k_loc.
            for j_loc in range(2):
                for k_loc in range(2):
                    local_b_idx = j_loc * 2 + k_loc
                    global_b_idx = (j_off + j_loc) * 4 + (k_off + k_loc)
                    V_full[global_b_idx, col] = V_22[local_b_idx, t]
            
            # C mapping: local row k_loc (0,1) -> global col k_off + k_loc. local col i_loc (0,1) -> global row i_loc.
            # C index is k*n + i.
            for k_loc in range(2):
                for i_loc in range(2):
                    local_c_idx = k_loc * 2 + i_loc
                    global_c_idx = (k_off + k_loc) * 2 + i_loc
                    W_full[global_c_idx, col] = W_22[local_c_idx, t]

    # Verify
    if verify(U_full, V_full, W_full):
        return to_lists(U_full, V_full, W_full, full_rank)
    
    # Fallback: Schoolbook
    def build_schoolbook():
        rank = n * m * p
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

    U, V, W = build_schoolbook()
    return to_lists(U, V, W, n * m * p)
# EVOLVE-BLOCK-END
