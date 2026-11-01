"""Matrix multiplication 2x4x4: rank-28 Strassen-based decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 tensor decomposition for 2x4 x 4x4 matrix multiplication
    using 4 Strassen 2x2x2 blocks, each of rank 7.
    
    U: shape (n*m, rank) = (8, 28)
    V: shape (m*p, rank) = (16, 28) 
    W: shape (p*n, rank) = (8, 28)
    
    Indexing:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    """
    n, m, p = 2, 4, 4
    
    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # For a 2x2 x 2x2 multiplication, the 7 Strassen products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each term contributes to U (A indices), V (B indices), W (C indices)
    # For a 2x2x2 block with local indices (i,j,k) in {0,1}^3:
    # U_local[i,j], V_local[j,k], W_local[k,i]
    
    # The 7 Strassen terms for one 2x2x2 block, each as (U_local, V_local, W_local)
    # U_local is 2x2 (rows=i, cols=j), V_local is 2x2 (rows=j, cols=k), W_local is 2x2 (rows=k, cols=i)
    
    def strassen_terms():
        """Return list of 7 (U2x2, V2x2, W2x2) terms for one 2x2x2 block."""
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C00(+), C11(+)
        # U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]]
        terms.append(([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]))
        
        # M2 = (A10 + A11)B00 -> contributes to C10(+)
        # U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,0],[1,0]]  (W[k,i]: k=1,i=0 -> W[1,0]=1)
        terms.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]))
        
        # M3 = A00(B10 - B11) -> contributes to C01(+)
        # U: [[1,0],[0,0]], V: [[0,0],[1,-1]], W: [[0,1],[0,0]]  (W[k,i]: k=0,i=1 -> W[0,1]=1)
        terms.append(([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]))
        
        # M4 = A11(B01 - B00) -> contributes to C00(+), C10(+)
        # U: [[0,0],[0,1]], V: [[-1,1],[0,0]], W: [[1,0],[1,0]]  (W[k,i]: W[0,0]=1, W[1,0]=1)
        terms.append(([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[1,0]]))
        
        # M5 = (A00 + A01)B11 -> contributes to C00(-), C01(+)
        # U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[-1,0],[0,1]]  (W[0,0]=-1, W[1,1]=1)
        terms.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]))
        
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C11(+)
        # U: [[-1,0],[1,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]]  (W[k,i]: k=1,i=1 -> W[1,1]=1)
        terms.append(([[ -1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]))
        
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C00(+)
        # U: [[0,1],[0,-1]], V: [[0,0],[1,1]], W: [[1,0],[0,0]]  (W[k,i]: k=0,i=0 -> W[0,0]=1)
        terms.append(([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]))
        
        return terms
    
    s_terms = strassen_terms()
    assert len(s_terms) == 7
    
    # Now build the full 2x4x4 decomposition from 4 Strassen blocks
    # Block (i_offset, j_offset, k_offset) where i_offset in {0,1}, j_offset in {0,2}, k_offset in {0,2}
    # For each block, local indices (i,j,k) in {0,1}^3 map to global:
    #   global_i = i_offset + i
    #   global_j = j_offset + j  
    #   global_k = k_offset + k
    
    all_terms = []
    for i_off in [0, 1]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                for (U_loc, V_loc, W_loc) in s_terms:
                    all_terms.append((i_off, j_off, k_off, U_loc, V_loc, W_loc))
    
    rank = len(all_terms)
    assert rank == 28
    
    # Build U (8 x 28), V (16 x 28), W (8 x 28)
    U = [[0] * rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0] * rank for _ in range(16)]  # m*p = 4*4 = 16
    W = [[0] * rank for _ in range(8)]   # p*n = 4*2 = 8
    
    for c, (i_off, j_off, k_off, U_loc, V_loc, W_loc) in enumerate(all_terms):
        # Fill U: U[i*m + j] where i = i_off + li, j = j_off + lj
        for li in range(2):
            for lj in range(2):
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][c] = U_loc[li][lj]
        
        # Fill V: V[j*p + k] where j = j_off + lj, k = k_off + lk
        for lj in range(2):
            for lk in range(2):
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][c] = V_loc[lj][lk]
        
        # Fill W: W[k*n + i] where k = k_off + lk, i = i_off + li
        for lk in range(2):
            for li in range(2):
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][c] = W_loc[lk][li]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
