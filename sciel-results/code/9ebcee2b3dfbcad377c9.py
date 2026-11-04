"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Partition the middle dimension m=3 into a 2x1 split.
    - For the 2x2 sub-block (cols 0,1 of A; rows 0,1 of B), apply Strassen's algorithm
      on 5 disjoint 2x2 blocks of the output (splitting the 10 columns into 5 pairs).
      This gives 5 * 7 = 35 rank terms.
    - For the remaining 1x1 sub-block (col 2 of A; row 2 of B), use schoolbook multiplication.
      This gives 2 * 10 = 20 rank terms.
    - Total rank: 35 + 20 = 55.
    
    Index conventions:
    - U[i, j] corresponds to A[i, j] with i in [0,2), j in [0,3). U shape: (6, rank)
    - V[j, k] corresponds to B[j, k] with j in [0,3), k in [0,10). V shape: (30, rank)
    - W[k, i] corresponds to C[i, k] with k in [0,10), i in [0,2). W shape: (20, rank)
      (trace convention: W[k*n + i, r])
    """
    n, m, p = 2, 3, 10
    rank = 55
    
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    idx = 0
    
    # Strassen decomposition for the 2x2 sub-block
    # A uses columns 0,1; B uses rows 0,1
    # Output C columns are split into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices: row * 3 + col
        a00 = 0 * 3 + 0  # 0
        a01 = 0 * 3 + 1  # 1
        a10 = 1 * 3 + 0  # 3
        a11 = 1 * 3 + 1  # 4
        
        # B indices: row * 10 + col
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices: k * 2 + i (trace convention)
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
    
    # Schoolbook for the 1x1 sub-block (A[:, 2] * B[2, :])
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
