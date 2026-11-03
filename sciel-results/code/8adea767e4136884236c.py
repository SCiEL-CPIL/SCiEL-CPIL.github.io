"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 * 2x2.
    Returns U, V, W each of shape (4, 7).
    U: 2x2 matrix A indices -> U[i*2+j, t]
    V: 2x2 matrix B indices -> V[j*2+k, t]
    W: 2x2 output C indices -> W[k*2+i, t]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12 - B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21 - B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds rank-49 decomposition for 4x4 * 4x4.
    Uses Strassen on 2x2 blocks.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U22, V22, W22 = _strassen_2x2x2()
    
    # Outer Strassen terms
    # Each term t_outer corresponds to a product of linear combinations of 2x2 blocks.
    # We define the coefficients for A_blocks and B_blocks for each of the 7 outer terms.
    # A_blocks: 00, 01, 10, 11 (row_block, col_block)
    # B_blocks: 00, 01, 10, 11 (row_block, col_block)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Coefficients for A blocks: dict (term, block_idx) -> coeff
    # Block indices: 0:00, 1:01, 2:10, 3:11
    a_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1},      # M7
    ]
    
    b_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]
    
    # W coefficients for the output blocks C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_block_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C00 from M1, M4, M5, M7
        {2: 1, 4: 1},                 # C01 from M3, M5
        {1: 1, 3: 1},                 # C10 from M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1},   # C11 from M1, M2, M3, M6
    ]
    
    # Helper to map block index to global indices
    def block_to_global(block_idx, local_i, local_j):
        rb = block_idx // 2
        cb = block_idx % 2
        gi = rb * 2 + local_i
        gj = cb * 2 + local_j
        return gi, gj

    # Iterate over outer terms
    for t_outer in range(7):
        # Iterate over inner Strassen terms
        for t_inner in range(7):
            t_global = t_outer * 7 + t_inner
            
            # Get inner coefficients for U22, V22, W22
            # U22 is 4x7, index [local_a_idx, t_inner]
            # local_a_idx is 0..3 for 2x2 block A_eff
            
            for la in range(4):
                u_val = U22[la][t_inner]
                if u_val != 0:
                    # la corresponds to (li, lj) in 2x2 block
                    li = la // 2
                    lj = la % 2
                    
                    # This u_val applies to all A_blocks that have non-zero coeff in this outer term
                    for a_block_idx, a_coeff in a_coeffs[t_outer].items():
                        if a_coeff == 0: continue
                        gi, gj = block_to_global(a_block_idx, li, lj)
                        U[gi * m + gj][t_global] += u_val * a_coeff
            
            for lb in range(4):
                v_val = V22[lb][t_inner]
                if v_val != 0:
                    li = lb // 2
                    lj = lb % 2
                    for b_block_idx, b_coeff in b_coeffs[t_outer].items():
                        if b_coeff == 0: continue
                        gi, gj = block_to_global(b_block_idx, li, lj)
                        V[gi * p + gj][t_global] += v_val * b_coeff
            
            for lc in range(4):
                w_val = W22[lc][t_inner]
                if w_val != 0:
                    li = lc // 2
                    lj = lc % 2
                    # W22 index lc corresponds to C_eff[li, lj] in 2x2 block
                    # The output block C_eff is one of C00, C01, C10, C11?
                    # No, W22 computes the 2x2 result of the "effective" multiplication.
                    # The "effective" multiplication result is added to specific C blocks.
                    # Wait, the Strassen algorithm computes the 4 blocks of C.
                    # The outer terms M1..M7 are intermediate 2x2 matrices.
                    # The final C blocks are linear combinations of M1..M7.
                    # So for each outer term t_outer, the inner Strassen computes M_t_outer (a 2x2 matrix).
                    # We need to distribute M_t_outer into the final C blocks.
                    
                    # M_t_outer contributes to C blocks based on w_block_coeffs.
                    # w_block_coeffs[c_block_idx] contains {t_outer: coeff}
                    # So for this t_outer, we look at which c_blocks it contributes to.
                    
                    # Invert w_block_coeffs to find contribution of t_outer to each c_block
                    for c_block_idx in range(4):
                        if t_outer in w_block_coeffs[c_block_idx]:
                            c_coeff = w_block_coeffs[c_block_idx][t_outer]
                            if c_coeff == 0: continue
                            gi, gj = block_to_global(c_block_idx, li, lj)
                            W[gj * n + gi][t_global] += w_val * c_coeff

    return U, V, W

def _build_4x4x3_schoolbook():
    """
    Builds rank-48 decomposition for 4x4 * 4x3.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, p_global=7):
    """
    Combines two decompositions.
    U1, V1, W1 are for 4x4x4 (using first 4 cols of B, output 4x4).
    U2, V2, W2 are for 4x4x3 (using last 3 cols of B, output 4x3).
    """
    n, m = 4, 4
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p_global)]
    W = [[0]*total_rank for _ in range(p_global*n)]
    
    # Copy part 1
    for i in range(n*m):
        for t in range(rank1):
            U[i][t] = U1[i][t]
    for j in range(m):
        for k in range(4): # First 4 cols of B
            for t in range(rank1):
                V[j*p_global + k][t] = V1[j*4 + k][t]
    for k in range(4): # First 4 cols of C
        for i in range(n):
            for t in range(rank1):
                W[k*n + i][t] = W1[k*n + i][t]
                
    # Copy part 2
    offset = rank1
    for i in range(n*m):
        for t in range(rank2):
            U[i][t + offset] = U2[i][t]
    for j in range(m):
        for k in range(3): # Last 3 cols of B (indices 4,5,6)
            for t in range(rank2):
                V[j*p_global + (4 + k)][t + offset] = V2[j*3 + k][t]
    for k in range(3): # Last 3 cols of C (indices 4,5,6)
        for i in range(n):
            for t in range(rank2):
                W[(4 + k)*n + i][t + offset] = W2[k*n + i][t]
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=4, m=4, p=7):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        a_t = 0.0
        for i in range(n):
            for j in range(m):
                a_t += U_arr[i * m + j, t] * A[i, j]
        b_t = 0.0
        for j in range(m):
            for k in range(p):
                b_t += V_arr[j * p + k, t] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k * n + i, t] * a_t * b_t
    
    diff = np.max(np.abs(C_computed - C_expected))
    return diff < 1e-10, diff

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Build 4x4x4 Strassen
    U1, V1, W1 = _build_4x4x4_strassen()
    
    # Build 4x4x3 Schoolbook
    U2, V2, W2 = _build_4x4x3_schoolbook()
    
    # Combine
    result = _combine_decompositions(U1, V1, W1, U2, V2, W2)
    
    # Verify
    valid, diff = _verify_decomposition(result["U"], result["V"], result["W"])
    
    if not valid:
        # Fallback to pure schoolbook 4x4x7
        n, m, p = 4, 4, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
