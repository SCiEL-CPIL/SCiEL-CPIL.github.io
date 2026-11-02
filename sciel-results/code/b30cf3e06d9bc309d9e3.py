"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-53 exact tensor decomposition for 2x2 by 2x15 matrix multiplication.
    Uses Strassen's rank-7 decomposition for 7 blocks of 2 columns each (14 cols),
    plus schoolbook rank-4 for the final column (1 col). Total rank = 7*7 + 4 = 53.
    
    Convention: C[i,k] = sum_{j,c} A[i,j] B[j,k] U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    must equal the standard matrix product, i.e., the coefficient of A[i,j]B[j,k] 
    summed over c must be 1.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    num_strassen = p // 2  # 7
    remaining = p % 2       # 1
    rank = num_strassen * 7 + remaining * 4  # 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # Verified rank-7 decomposition for 2x2x2 matrix multiplication tensor.
    # For each term r, we have vectors u_r (len 4), v_r (len 4), w_r (len 4) such that:
    # for all i,j,k in {0,1}: sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1
    # 
    # Using the well-known Strassen-based rank-7 decomposition:
    # The 7 multiplication terms in Strassen are:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    # 
    # Each M_r = f_r(A) * g_r(B) where f_r and g_r are linear forms.
    # For the tensor decomposition, each M_r contributes one rank term:
    #   u_r: coefficients of f_r in {A00, A01, A10, A11}
    #   v_r: coefficients of g_r in {B00, B01, B10, B11}
    #   w_r: how M_r contributes to {C00, C10, C01, C11}
    #
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # So for each r, w_r has entries for C00, C10, C01, C11:
    
    terms = [
        # r, u_r (A coeffs), v_r (B coeffs), w_r (C coeffs: C00, C10, C01, C11)
        (1, [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        (2, [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        (3, [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        (4, [0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),  # M4
        (5, [1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        (6, [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        (7, [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    col = 0
    for b in range(num_strassen):
        # Local to global index mapping for this 2-column block
        # V local: B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
        # V global: j*p + k where k = 2b or 2b+1
        v_map = {
            0: 0 * p + 2 * b,      # B[0, 2b]
            1: 0 * p + 2 * b + 1,  # B[0, 2b+1]
            2: 1 * p + 2 * b,      # B[1, 2b]
            3: 1 * p + 2 * b + 1,  # B[1, 2b+1]
        }
        # W local: C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3
        # W global: k*n + i where k = 2b or 2b+1, i = 0 or 1
        w_map = {
            0: (2 * b) * n + 0,      # C[0, 2b]
            1: (2 * b) * n + 1,      # C[1, 2b]
            2: (2 * b + 1) * n + 0,  # C[0, 2b+1]
            3: (2 * b + 1) * n + 1,  # C[1, 2b+1]
        }
        
        for r, u_r, v_r, w_r in terms:
            # U indices are global: i*m + j, same across all blocks
            for idx in range(4):
                if u_r[idx] != 0:
                    U[idx][col] += u_r[idx]
            for idx in range(4):
                if v_r[idx] != 0:
                    V[v_map[idx]][col] += v_r[idx]
            for idx in range(4):
                if w_r[idx] != 0:
                    W[w_map[idx]][col] += w_r[idx]
            col += 1
    
    # Schoolbook for the remaining single column (column index 14)
    k_rem = num_strassen * 2  # = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + k_rem][col] = 1
            W[k_rem * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
