"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy: Use a block-structured approach.
    - The 3x4 * 4x4 product can be decomposed into blocks.
    - We use the known rank-19 Bini-Capovani-De Falco-Ladisi (BCDFL) decomposition for the 3x3x3 core.
    - For the remaining parts (3x1 * 4x4 and 1x4 * 4x4), we use structured decompositions.
    
    Actually, let's use a simpler and more reliable approach:
    - Use Strassen-like block decomposition where possible.
    - The 3x4 * 4x4 can be split as:
      A (3x4) = [A1 (3x2), A2 (3x2)]
      B (4x4) = [[B11 (2x2), B12 (2x2)],
                 [B21 (2x2), B22 (2x2)]]
      C = A1*B1 + A2*B2 where B1 = [B11, B21] (4x2) and B2 = [B12, B22] (4x2)
    
    This gives two subproblems: (3x2)*(4x2) -> 3x2 and (3x2)*(4x2) -> 3x2.
    Each (3x2)*(4x2) can be further decomposed.
    
    Let me use a direct rank-44 construction based on Strassen for 2x2x2 blocks:
    - Split 3x4 * 4x4 into blocks of sizes that allow Strassen.
    - 3 = 2 + 1, 4 = 2 + 2
    - This gives 8 blocks, some 2x2x2 (use Strassen rank 7) and some 1x2x2 (use schoolbook rank 2).
    
    Actually, let's just use a known good construction. The schoolbook rank is 3*4*4 = 48.
    We can do better by using Strassen on the 2x2x2 sub-blocks.
    
    Let me construct this carefully:
    A is 3x4, B is 4x4, C is 3x4.
    
    Partition A into rows: A = [A_top (2x4); A_bottom (1x4)]
    Partition B into columns: B = [B_left (4x2), B_right (4x2)]
    Then C = [A_top*B_left (2x2), A_top*B_right (2x2); A_bottom*B_left (1x2), A_bottom*B_right (1x2)]
    
    So we have 4 subproblems:
    1. (2x4)*(4x2) -> 2x2: rank can be 7*2 = 14 (two Strassen 2x2x2) or schoolbook 2*4*2=16
       Actually, (2x4)*(4x2) = sum over k of A[i,k]*B[k,j]. We can split the 4 dimension:
       A (2x4) = [A1 (2x2), A2 (2x2)], B (4x2) = [B1 (2x2); B2 (2x2)]
       Result = A1*B1 + A2*B2, each (2x2)*(2x2) -> 2x2, rank 7 each, total 14.
    2. (2x4)*(4x2) -> 2x2: same as above, rank 14.
    3. (1x4)*(4x2) -> 1x2: schoolbook rank 1*4*2 = 8.
    4. (1x4)*(4x2) -> 1x2: schoolbook rank 8.
    
    Total rank = 14 + 14 + 8 + 8 = 44.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using the block decomposition above.
    
    # Let's index: A[i,j] with i in [0,3), j in [0,4)
    # B[j,k] with j in [0,4), k in [0,4)
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # U[i*m + j, c] = U[i*4 + j, c]
    # V[j*p + k, c] = V[j*4 + k, c]
    # W[k*n + i, c] = W[k*3 + i, c]
    
    # Block 1: A_top (2x4) * B_left (4x2) -> C_top_left (2x2)
    #   A_top[i,j] for i in [0,2), j in [0,4)
    #   B_left[j,k] for j in [0,4), k in [0,2)
    #   C_top_left[i,k] for i in [0,2), k in [0,2)
    #   Split j into j1 in [0,2) and j2 in [2,4)
    #   A_top = [A1 (2x2), A2 (2x2)] where A1[i,j1]=A[i,j1], A2[i,j2-2]=A[i,j2]
    #   B_left = [B1 (2x2); B2 (2x2)] where B1[j1,k]=B[j1,k], B2[j2-2,k]=B[j2,k]
    #   C_top_left = A1*B1 + A2*B2
    #   Use Strassen for each 2x2x2.
    
    # Strassen for 2x2x2:
    #   X1 = (A00 + A11)(B00 + B11)
    #   X2 = (A10 + A11)B00
    #   X3 = A00(B10 - B00)
    #   X4 = A11(B01 - B10)
    #   X5 = (A00 + A01)B11
    #   X6 = (A10 - A00)(B00 + B10)
    #   X7 = (A01 - A11)(B01 + B11)
    #   C00 = X1 + X4 - X5 + X7
    #   C01 = X3 + X5
    #   C10 = X2 + X4
    #   C11 = X1 - X2 + X3 + X6
    
    def strassen_2x2x2(U_base, V_base, W_base, i_off, j_off, k_off, rank_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        U rows: i_off..i_off+1, j_off..j_off+1 -> U[i*4+j]
        V rows: j_off..j_off+1, k_off..k_off+1 -> V[j*4+k]
        W rows: k_off..k_off+1, i_off..i_off+1 -> W[k*3+i]
        """
        c = rank_start
        # X1 = (A00 + A11)(B00 + B11)
        U[i_off*4 + j_off][c] = 1
        U[(i_off+1)*4 + (j_off+1)][c] = 1
        V[j_off*4 + k_off][c] = 1
        V[(j_off+1)*4 + (k_off+1)][c] = 1
        W[k_off*3 + i_off][c] = 1
        W[(k_off+1)*3 + (i_off+1)][c] = 1
        W[k_off*3 + (i_off+1)][c] = 1
        W[(k_off+1)*3 + i_off][c] = 1
        c += 1
        
        # X2 = (A10 + A11)B00
        U[(i_off+1)*4 + j_off][c] = 1
        U[(i_off+1)*4 + (j_off+1)][c] = 1
        V[j_off*4 + k_off][c] = 1
        W[k_off*3 + (i_off+1)][c] = 1
        c += 1
        
        # X3 = A00(B10 - B00)
        U[i_off*4 + j_off][c] = 1
        V[(j_off+1)*4 + k_off][c] = 1
        V[j_off*4 + k_off][c] = -1
        W[k_off*3 + i_off][c] = 1
        W[(k_off+1)*3 + i_off][c] = 1
        c += 1
        
        # X4 = A11(B01 - B10)
        U[(i_off+1)*4 + (j_off+1)][c] = 1
        V[j_off*4 + (k_off+1)][c] = 1
        V[(j_off+1)*4 + k_off][c] = -1
        W[(k_off+1)*3 + (i_off+1)][c] = 1
        W[k_off*3 + (i_off+1)][c] = 1
        c += 1
        
        # X5 = (A00 + A01)B11
        U[i_off*4 + j_off][c] = 1
        U[i_off*4 + (j_off+1)][c] = 1
        V[(j_off+1)*4 + (k_off+1)][c] = 1
        W[(k_off+1)*3 + i_off][c] = 1
        W[(k_off+1)*3 + (i_off+1)][c] = 1
        c += 1
        
        # X6 = (A10 - A00)(B00 + B10)
        U[(i_off+1)*4 + j_off][c] = 1
        U[i_off*4 + j_off][c] = -1
        V[j_off*4 + k_off][c] = 1
        V[(j_off+1)*4 + k_off][c] = 1
        W[k_off*3 + (i_off+1)][c] = 1
        c += 1
        
        # X7 = (A01 - A11)(B01 + B11)
        U[i_off*4 + (j_off+1)][c] = 1
        U[(i_off+1)*4 + (j_off+1)][c] = -1
        V[j_off*4 + (k_off+1)][c] = 1
        V[(j_off+1)*4 + (k_off+1)][c] = 1
        W[(k_off+1)*3 + i_off][c] = 1
        c += 1
        
        return c
    
    def schoolbook_1x2x2(U_base, V_base, W_base, i_idx, j_off, k_off, rank_start):
        """
        Schoolbook for (1x2)*(2x2) -> 1x2.
        U row: i_idx*4 + j for j in [j_off, j_off+2)
        V rows: (j_off+k)*4 + (k_off+l) for k in [0,2), l in [0,2)
        W rows: (k_off+l)*3 + i_idx for l in [0,2)
        """
        c = rank_start
        for k in range(2):
            for l in range(2):
                U[i_idx*4 + (j_off + k)][c] = 1
                V[(j_off + k)*4 + (k_off + l)][c] = 1
                W[(k_off + l)*3 + i_idx][c] = 1
                c += 1
        return c
    
    # Total rank estimate: 7+7+7+7 (four Strassen) + 8+8 (two schoolbook 1x2x2) = 28+16 = 44
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 3*4 = 12 rows
    V = [[0]*rank for _ in range(m*p)]  # 4*4 = 16 rows
    W = [[0]*rank for _ in range(p*n)]  # 4*3 = 12 rows
    
    c = 0
    
    # Block 1: A_top (2x4) * B_left (4x2) -> C_top_left (2x2)
    #   A1 (2x2): i in [0,2), j in [0,2)
    #   B1 (2x2): j in [0,2), k in [0,2)
    c = strassen_2x2x2(U, V, W, 0, 0, 0, c)  # A1*B1
    #   A2 (2x2): i in [0,2), j in [2,4)
    #   B2 (2x2): j in [2,4), k in [0,2)
    c = strassen_2x2x2(U, V, W, 0, 2, 0, c)  # A2*B2
    
    # Block 2: A_top (2x4) * B_right (4x2) -> C_top_right (2x2)
    #   A1 (2x2): i in [0,2), j in [0,2)
    #   B1 (2x2): j in [0,2), k in [2,4)
    c = strassen_2x2x2(U, V, W, 0, 0, 2, c)  # A1*B1
    #   A2 (2x2): i in [0,2), j in [2,4)
    #   B2 (2x2): j in [2,4), k in [2,4)
    c = strassen_2x2x2(U, V, W, 0, 2, 2, c)  # A2*B2
    
    # Block 3: A_bottom (1x4) * B_left (4x2) -> C_bottom_left (1x2)
    c = schoolbook_1x2x2(U, V, W, 2, 0, 0, c)
    
    # Block 4: A_bottom (1x4) * B_right (4x2) -> C_bottom_right (1x2)
    c = schoolbook_1x2x2(U, V, W, 2, 0, 2, c)
    
    # Verify c == rank
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Should not happen, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
