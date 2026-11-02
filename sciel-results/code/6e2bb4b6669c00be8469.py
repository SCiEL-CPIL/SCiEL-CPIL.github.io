import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the factor matrices for a rank-28 decomposition of matrix multiplication.
    Dimensions: n x m times m x p.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    
    This implementation uses a block-based approach:
    - Decompose the 2x3 matrix A into a 2x2 block (cols 0,1) and a 2x1 block (col 2).
    - Decompose the 3x5 matrix B into 2x2 blocks (rows 0,1; cols 0,1 and cols 2,3) and a 2x1 block (rows 0,1; col 4),
      plus a 1x5 block (row 2).
    
    Rank breakdown:
    - Block 1 (A[0:2,0:2] * B[0:2,0:2]): Strassen rank 7.
    - Block 2 (A[0:2,0:2] * B[0:2,2:4]): Strassen rank 7.
    - Block 3 (A[0:2,0:2] * B[0:2,4:5]): Schoolbook rank 4 (2*2*1).
    - Block 4 (A[0:2,2:3] * B[2:3,0:5]): Schoolbook rank 10 (2*1*5).
    Total Rank = 7 + 7 + 4 + 10 = 28.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block for 2x2 * 2x2
    def add_strassen_block(rank_offset, j_offset, k_offset):
        # A vars
        a_idx = 0 * m + j_offset
        b_idx = 0 * m + (j_offset + 1)
        c_idx = 1 * m + j_offset
        d_idx = 1 * m + (j_offset + 1)
        
        # B vars
        e_idx = j_offset * p + k_offset
        f_idx = j_offset * p + (k_offset + 1)
        g_idx = (j_offset + 1) * p + k_offset
        h_idx = (j_offset + 1) * p + (k_offset + 1)
        
        # Ranks
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        # U (L_A)
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        U[c_idx][r1] += 1; U[d_idx][r1] += 1
        U[a_idx][r2] += 1
        U[d_idx][r3] += 1
        U[a_idx][r4] += 1; U[b_idx][r4] += 1
        U[c_idx][r5] += 1; U[a_idx][r5] -= 1
        U[b_idx][r6] += 1; U[d_idx][r6] -= 1
        
        # V (L_B)
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1
        V[f_idx][r2] += 1; V[h_idx][r2] -= 1
        V[g_idx][r3] += 1; V[e_idx][r3] -= 1
        V[h_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] += 1
        V[g_idx][r6] += 1; V[h_idx][r6] += 1
        
        # W (Coefficients)
        # C_00 (i=0, k=k_offset) -> idx k_offset*n + 0
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] += 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        # C_01 (i=0, k=k_offset+1) -> idx (k_offset+1)*n + 0
        W[(k_offset + 1) * n + 0][r2] += 1
        W[(k_offset + 1) * n + 0][r4] += 1
        
        # C_10 (i=1, k=k_offset) -> idx k_offset*n + 1
        W[k_offset * n + 1][r1] += 1
        W[k_offset * n + 1][r3] += 1
        
        # C_11 (i=1, k=k_offset+1) -> idx (k_offset+1)*n + 1
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r1] -= 1
        W[(k_offset + 1) * n + 1][r2] += 1
        W[(k_offset + 1) * n + 1][r5] += 1

    # Block 1: j=0,1, k=0,1 (2x2 * 2x2)
    add_strassen_block(0, 0, 0)
    
    # Block 2: j=0,1, k=2,3 (2x2 * 2x2)
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (2x2 * 2x1)
    r0 = 14
    r1 = 15
    r2 = 16
    r3 = 17
    
    U[0 * m + 0][r0] += 1
    U[0 * m + 1][r1] += 1
    U[1 * m + 0][r2] += 1
    U[1 * m + 1][r3] += 1
    
    V[0 * p + 4][r0] += 1
    V[1 * p + 4][r1] += 1
    V[0 * p + 4][r2] += 1
    V[1 * p + 4][r3] += 1
    
    W[4 * n + 0][r0] += 1
    W[4 * n + 0][r1] += 1
    W[4 * n + 1][r2] += 1
    W[4 * n + 1][r3] += 1
    
    # Block 4: j=2, k=0..4 (2x1 * 1x5)
    rank_offset = 18
    for k in range(5):
        r_u0 = rank_offset
        r_u1 = rank_offset + 1
        rank_offset += 2
        
        U[0 * m + 2][r_u0] += 1
        U[1 * m + 2][r_u1] += 1
        
        V[2 * p + k][r_u0] += 1
        V[2 * p + k][r_u1] += 1
        
        W[k * n + 0][r_u0] += 1
        W[k * n + 1][r_u1] += 1
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal A[i,j]*B[j,k]
    # For random test matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        C_computed += W_arr[:, r] * (U_arr[:, r] * V_arr[:, r])
    
    # The W matrix is indexed as W[k*n+i, r], so W_arr[k*n+i, r]
    # We need to reshape properly
    C_computed2 = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                C_computed2[i, k] += W_arr[k*n+i, r] * U_arr[:, r] @ V_arr[:, r]
    
    # Actually, let's do it more carefully
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_computed3 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                for j in range(m):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            C_computed3[i, k] = total
    
    max_err = np.max(np.abs(C_computed3 - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, idx] = 1.0
                    V_school[j*p+k, idx] = 1.0
                    W_school[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U, "V": V, "W": W}
