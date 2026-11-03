"""Matrix multiplication 3x4x4 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_strassen_2x2():
    """
    Returns the rank-7 decomposition for 2x2 matrix multiplication.
    U has shape (4, 7) for A elements indexed as A[0,0], A[0,1], A[1,0], A[1,1]
    V has shape (4, 7) for B elements indexed as B[0,0], B[0,1], B[1,0], B[1,1]
    W has shape (4, 7) for C elements indexed as C[0,0], C[0,1], C[1,0], C[1,1]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A22(B12 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C12 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    
    # C21 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    
    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _build_rank44():
    """
    Build a rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split A (3x4) into A_top (2x4) and A_bot (1x4)
    - C_bot = A_bot * B: schoolbook rank 16
    - C_top = A_top * B: rank 28 using Strassen-based approach
    
    For C_top (2x4 * 4x4):
    Split A_top = [A_L, A_R] where A_L, A_R are 2x2
    Split B = [[B_00, B_01], [B_10, B_11]] where B_ij are 2x2
    C_top = [A_L*B_00 + A_R*B_10, A_L*B_01 + A_R*B_11]
    
    We use Strassen's algorithm on a 4x4 problem by padding:
    Pad A_top to 4x4: A_pad = [[A_L, A_R], [0, 0]]
    B is already 4x4: B = [[B_00, B_01], [B_10, B_11]]
    Product C_pad = A_pad * B
    The top-left 2x4 block of C_pad is exactly C_top.
    
    For 4x4 Strassen, we use the recursive structure with rank 28.
    Actually, let me use a direct rank-28 construction for 2x4x4.
    
    Alternative approach: Use 4 independent 2x2 Strassen multiplications.
    But that gives rank 4*7 = 28 only if they are independent.
    
    Let me think more carefully. For 2x4 * 4x4:
    A_top = [A1, A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x4 (each split into two 2x2 blocks)
    
    Actually, let me use a cleaner approach:
    A_top (2x4) * B (4x4) = C_top (2x4)
    
    Write A_top = [A_L | A_R] where A_L, A_R are 2x2
    Write B = [[B_00 | B_01], [B_10 | B_11]] where each B_ij is 2x2
    
    C_top = [A_L*B_00 + A_R*B_10 | A_L*B_01 + A_R*B_11]
    
    This requires computing:
    P1 = A_L * B_00
    P2 = A_R * B_10
    P3 = A_L * B_01
    P4 = A_R * B_11
    
    Each Pi is a 2x2 * 2x2 multiplication, rank 7 each.
    Total rank = 4 * 7 = 28.
    
    But wait, we need to combine them:
    C_top[:, 0:2] = P1 + P2
    C_top[:, 2:4] = P3 + P4
    
    So we need 4 independent 2x2 Strassen decompositions.
    Total rank for C_top = 28.
    
    For C_bot (1x4 * 4x4 = 1x4):
    Schoolbook rank = 1*4*4 = 16.
    
    Total rank = 28 + 16 = 44.
    """
    n, m, p = 3, 4, 4
    
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    # --- Part 1: C_top = A_top * B (rank 28) ---
    # A_top rows: i=0,1 (indices 0,1 in A)
    # B: full 4x4
    # C_top: rows i=0,1, cols k=0,1,2,3
    
    # Split A_top = [A_L | A_R]
    # A_L = A[0:2, 0:2] (rows 0,1; cols 0,1)
    # A_R = A[0:2, 2:4] (rows 0,1; cols 2,3)
    
    # Split B into 2x2 blocks:
    # B_00 = B[0:2, 0:2], B_01 = B[0:2, 2:4]
    # B_10 = B[2:4, 0:2], B_11 = B[2:4, 2:4]
    
    # C_top[:, 0:2] = A_L*B_00 + A_R*B_10
    # C_top[:, 2:4] = A_L*B_01 + A_R*B_11
    
    # We need 4 Strassen decompositions:
    # S1: A_L * B_00 -> contributes to C_top[:, 0:2]
    # S2: A_R * B_10 -> contributes to C_top[:, 0:2]
    # S3: A_L * B_01 -> contributes to C_top[:, 2:4]
    # S4: A_R * B_11 -> contributes to C_top[:, 2:4]
    
    U_str, V_str, W_str = _build_strassen_2x2()
    
    # Helper: A 2x2 matrix with elements at global A indices
    # For A_L: elements are A[0,0], A[0,1], A[1,0], A[1,1] -> global indices 0,1,4,5
    # For A_R: elements are A[0,2], A[0,3], A[1,2], A[1,3] -> global indices 2,3,6,7
    
    # For B blocks (B is 4x4, indexed as B[j,k] with global index j*4+k):
    # B_00: B[0,0], B[0,1], B[1,0], B[1,1] -> global indices 0,1,4,5
    # B_01: B[0,2], B[0,3], B[1,2], B[1,3] -> global indices 2,3,6,7
    # B_10: B[2,0], B[2,1], B[3,0], B[3,1] -> global indices 8,9,12,13
    # B_11: B[2,2], B[2,3], B[3,2], B[3,3] -> global indices 10,11,14,15
    
    # For C_top (2x4, indexed as C[i,k] with global index k*3+i):
    # C_top[:, 0:2]: C[0,0], C[1,0], C[0,1], C[1,1] -> global indices 0,3,1,4
    # C_top[:, 2:4]: C[0,2], C[1,2], C[0,3], C[1,3] -> global indices 6,9,7,10
    
    def add_strassen(col_offset, A_indices, B_indices, C_indices):
        """Add a Strassen 2x2 decomposition starting at col_offset.
        A_indices: list of 4 global U indices for [A11, A12, A21, A22]
        B_indices: list of 4 global V indices for [B11, B12, B21, B22]
        C_indices: list of 4 global W indices for [C11, C12, C21, C22]
        """
        for r in range(7):
            col = col_offset + r
            for a_local, a_global in enumerate(A_indices):
                U[a_global, col] = U_str[a_local, r]
            for b_local, b_global in enumerate(B_indices):
                V[b_global, col] = V_str[b_local, r]
            for c_local, c_global in enumerate(C_indices):
                W[c_global, col] += W_str[c_local, r]
    
    # A_L indices: [A[0,0], A[0,1], A[1,0], A[1,1]] = [0, 1, 4, 5]
    A_L_idx = [0, 1, 4, 5]
    # A_R indices: [A[0,2], A[0,3], A[1,2], A[1,3]] = [2, 3, 6, 7]
    A_R_idx = [2, 3, 6, 7]
    
    # B_00 indices: [B[0,0], B[0,1], B[1,0], B[1,1]] = [0, 1, 4, 5]
    B_00_idx = [0, 1, 4, 5]
    # B_01 indices: [B[0,2], B[0,3], B[1,2], B[1,3]] = [2, 3, 6, 7]
    B_01_idx = [2, 3, 6, 7]
    # B_10 indices: [B[2,0], B[2,1], B[3,0], B[3,1]] = [8, 9, 12, 13]
    B_10_idx = [8, 9, 12, 13]
    # B_11 indices: [B[2,2], B[2,3], B[3,2], B[3,3]] = [10, 11, 14, 15]
    B_11_idx = [10, 11, 14, 15]
    
    # C_top left half (cols 0,1): [C[0,0], C[0,1], C[1,0], C[1,1]]
    # Global W index for C[i,k] = k*3 + i
    # C[0,0] -> 0*3+0 = 0
    # C[0,1] -> 1*3+0 = 3
    # C[1,0] -> 0*3+1 = 1
    # C[1,1] -> 1*3+1 = 4
    C_top_L_idx = [0, 3, 1, 4]  # [C11, C12, C21, C22] in Strassen convention
    
    # C_top right half (cols 2,3): [C[0,2], C[0,3], C[1,2], C[1,3]]
    # C[0,2] -> 2*3+0 = 6
    # C[0,3] -> 3*3+0 = 9
    # C[1,2] -> 2*3+1 = 7
    # C[1,3] -> 3*3+1 = 10
    C_top_R_idx = [6, 9, 7, 10]  # [C11, C12, C21, C22] in Strassen convention
    
    # S1: A_L * B_00 -> C_top[:, 0:2]
    add_strassen(0, A_L_idx, B_00_idx, C_top_L_idx)
    
    # S2: A_R * B_10 -> C_top[:, 0:2]
    add_strassen(7, A_R_idx, B_10_idx, C_top_L_idx)
    
    # S3: A_L * B_01 -> C_top[:, 2:4]
    add_strassen(14, A_L_idx, B_01_idx, C_top_R_idx)
    
    # S4: A_R * B_11 -> C_top[:, 2:4]
    add_strassen(21, A_R_idx, B_11_idx, C_top_R_idx)
    
    # --- Part 2: C_bot = A_bot * B (rank 16, schoolbook) ---
    # A_bot: row i=2, cols j=0,1,2,3
    # B: full 4x4
    # C_bot: row i=2, cols k=0,1,2,3
    
    col = 28
    for j in range(4):
        for k in range(4):
            U[2*m + j, col] = 1  # A[2, j]
            V[j*p + k, col] = 1  # B[j, k]
            W[k*n + 2, col] = 1  # C[2, k]
            col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank 44 first
    try:
        U, V, W = _build_rank44()
        
        # Verify the decomposition
        # Check all non-zero tensor entries
        nz_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    nz_indices.append((i*m+j, j*p+k, k*n+i))
        
        max_err = 0.0
        for (u_idx, v_idx, w_idx) in nz_indices:
            val = 0.0
            for r in range(44):
                val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            err = abs(val - 1.0)
            max_err = max(max_err, err)
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, fallback to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
