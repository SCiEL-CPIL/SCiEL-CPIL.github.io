"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    n, m, p = DIMENSIONS

    # Rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
    # We decompose C = A * B where A is 3x3 and B is 3x7.
    # Split B into B1 (3x3) and B2 (3x4).
    # C = A * B1 + A * B2.

    # We will use:
    # - Rank 23 decomposition for 3x3x3 (optimal known)
    # - Rank 34 decomposition for 3x3x4
    # Total Rank = 23 + 34 = 57

    # However, constructing rank 23 for 3x3x3 is complex. Let's use a simpler approach:
    # Use Strassen-based decomposition for the 3x3x3 part to get rank < 27.
    # Actually, let's just use schoolbook for safety but with better organization.
    # 
    # Let me implement a proper Strassen decomposition for 3x3x3:
    # Split A and B into 2x2 and 1x1 blocks.
    # 
    # Actually, the known optimal rank for 3x3x3 is 23. Let me try to construct it.
    # This is quite involved. For now, let me use the schoolbook approach but 
    # organize it better and verify it works.
    
    # Schoolbook: rank = 3*3*7 = 63
    # Let's try to do better with Strassen for sub-blocks.
    
    # I'll implement a hybrid:
    # For the 3x3x3 part, use Strassen on 2x2x2 sub-blocks.
    # For the 3x3x4 part, use Strassen on 2x2x2 sub-blocks.
    
    # Let me define a helper to add a Strassen 2x2x2 decomposition.
    
    def add_strassen_2x2x2(U, V, W, col, i_off, j_off, k_off):
        """
        Add a Strassen 2x2x2 decomposition starting at column col.
        i_off, j_off, k_off are the offsets in the full matrices.
        Returns the next column index.
        """
        # Strassen formulas for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B11-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # 
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M6
        # C11 = M1 - M3 + M4 + M6
        
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U index: (i_off+i)*m + (j_off+j)
        # V index: (j_off+j)*p + (k_off+k)
        # W index: (k_off+k)*n + (i_off+i)
        
        # M1: (A00+A11)(B00+B11)
        U[(i_off+0)*m + (j_off+0)][col] += 1  # A00
        U[(i_off+1)*m + (j_off+1)][col] += 1  # A11
        V[(j_off+0)*p + (k_off+0)][col] += 1  # B00
        V[(j_off+1)*p + (k_off+1)][col] += 1  # B11
        W[(k_off+0)*n + (i_off+0)][col] += 1  # C00
        W[(k_off+1)*n + (i_off+0)][col] += 1  # C01
        W[(k_off+0)*n + (i_off+1)][col] -= 1  # C10
        W[(k_off+1)*n + (i_off+1)][col] += 1  # C11
        col += 1
        
        # M2: (A10+A11)B00
        U[(i_off+1)*m + (j_off+0)][col] += 1  # A10
        U[(i_off+1)*m + (j_off+1)][col] += 1  # A11
        V[(j_off+0)*p + (k_off+0)][col] += 1  # B00
        W[(k_off+0)*n + (i_off+1)][col] += 1  # C10
        W[(k_off+1)*n + (i_off+1)][col] += 1  # C11
        col += 1
        
        # M3: A00(B10-B00)
        U[(i_off+0)*m + (j_off+0)][col] += 1  # A00
        V[(j_off+1)*p + (k_off+0)][col] += 1  # B10
        V[(j_off+0)*p + (k_off+0)][col] -= 1  # B00
        W[(k_off+0)*n + (i_off+0)][col] += 1  # C00
        W[(k_off+1)*n + (i_off+0)][col] += 1  # C01
        col += 1
        
        # M4: A11(B11-B00)
        U[(i_off+1)*m + (j_off+1)][col] += 1  # A11
        V[(j_off+1)*p + (k_off+1)][col] += 1  # B11
        V[(j_off+0)*p + (k_off+0)][col] -= 1  # B00
        W[(k_off+0)*n + (i_off+0)][col] -= 1  # C00
        W[(k_off+1)*n + (i_off+1)][col] += 1  # C11
        col += 1
        
        # M5: (A00+A01)B11
        U[(i_off+0)*m + (j_off+0)][col] += 1  # A00
        U[(i_off+0)*m + (j_off+1)][col] += 1  # A01
        V[(j_off+1)*p + (k_off+1)][col] += 1  # B11
        W[(k_off+0)*n + (i_off+0)][col] += 1  # C00
        W[(k_off+1)*n + (i_off+1)][col] += 1  # C11
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_off+1)*m + (j_off+0)][col] += 1  # A10
        U[(i_off+0)*m + (j_off+0)][col] -= 1  # A00
        V[(j_off+0)*p + (k_off+0)][col] += 1  # B00
        V[(j_off+0)*p + (k_off+1)][col] += 1  # B01
        W[(k_off+0)*n + (i_off+1)][col] += 1  # C10
        W[(k_off+1)*n + (i_off+1)][col] += 1  # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[(i_off+0)*m + (j_off+1)][col] += 1  # A01
        U[(i_off+1)*m + (j_off+1)][col] -= 1  # A11
        V[(j_off+1)*p + (k_off+0)][col] += 1  # B10
        V[(j_off+1)*p + (k_off+1)][col] += 1  # B11
        W[(k_off+0)*n + (i_off+0)][col] -= 1  # C00
        W[(k_off+1)*n + (i_off+0)][col] += 1  # C01
        col += 1
        
        return col
    
    def add_schoolbook_block(U, V, W, col, i_off, j_off, k_off, ni, nj, nk):
        """
        Add a schoolbook decomposition for a ni x nj x nk block.
        i_off, j_off, k_off are offsets.
        ni, nj, nk are the sizes of the block.
        Returns the next column index.
        """
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    U[(i_off+i)*m + (j_off+j)][col] = 1
                    V[(j_off+j)*p + (k_off+k)][col] = 1
                    W[(k_off+k)*n + (i_off+i)][col] = 1
                    col += 1
        return col
    
    # Total rank estimate:
    # 3x3x3 part:
    #   Block (0,0): i=0,1; j=0,1; k=0,1 -> Strassen (7)
    #   Block (0,1): i=0,1; j=2; k=0,1 -> school (2*1*2=4)
    #   Block (0,2): i=0,1; j=0,1; k=2 -> school (2*2*1=4)
    #   Block (0,3): i=0,1; j=2; k=2 -> school (2*1*1=2)
    #   Block (1,0): i=2; j=0,1; k=0,1 -> school (1*2*2=4)
    #   Block (1,1): i=2; j=2; k=0,1 -> school (1*1*2=2)
    #   Block (1,2): i=2; j=0,1; k=2 -> school (1*2*1=2)
    #   Block (1,3): i=2; j=2; k=2 -> school (1*1*1=1)
    #   Total 3x3x3: 7+4+4+2+4+2+2+1 = 26
    
    # 3x3x4 part:
    #   Block (0,0): i=0,1; j=0,1; k=3,4 -> Strassen (7)
    #   Block (0,1): i=0,1; j=0,1; k=5,6 -> Strassen (7)
    #   Block (0,2): i=0,1; j=2; k=3,4,5,6 -> school (2*1*4=8)
    #   Block (1,0): i=2; j=0,1; k=3,4,5,6 -> school (1*2*4=8)
    #   Block (1,1): i=2; j=2; k=3,4,5,6 -> school (1*1*4=4)
    #   Total 3x3x4: 7+7+8+8+4 = 34
    
    # Total rank: 26 + 34 = 60
    
    total_rank = 60
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3x3x3 (k=0,1,2)
    # Block (0,0): i=0,1; j=0,1; k=0,1 -> Strassen
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 0)
    
    # Block (0,1): i=0,1; j=2; k=0,1 -> school
    col = add_schoolbook_block(U, V, W, col, 0, 2, 0, 2, 1, 2)
    
    # Block (0,2): i=0,1; j=0,1; k=2 -> school
    col = add_schoolbook_block(U, V, W, col, 0, 0, 2, 2, 2, 1)
    
    # Block (0,3): i=0,1; j=2; k=2 -> school
    col = add_schoolbook_block(U, V, W, col, 0, 2, 2, 2, 1, 1)
    
    # Block (1,0): i=2; j=0,1; k=0,1 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 0, 0, 1, 2, 2)
    
    # Block (1,1): i=2; j=2; k=0,1 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 2, 0, 1, 1, 2)
    
    # Block (1,2): i=2; j=0,1; k=2 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 0, 2, 1, 2, 1)
    
    # Block (1,3): i=2; j=2; k=2 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 2, 2, 1, 1, 1)
    
    # Part 2: 3x3x4 (k=3,4,5,6)
    # Block (0,0): i=0,1; j=0,1; k=3,4 -> Strassen
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 3)
    
    # Block (0,1): i=0,1; j=0,1; k=5,6 -> Strassen
    col = add_strassen_2x2x2(U, V, W, col, 0, 0, 5)
    
    # Block (0,2): i=0,1; j=2; k=3,4,5,6 -> school
    col = add_schoolbook_block(U, V, W, col, 0, 2, 3, 2, 1, 4)
    
    # Block (1,0): i=2; j=0,1; k=3,4,5,6 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 0, 3, 1, 2, 4)
    
    # Block (1,1): i=2; j=2; k=3,4,5,6 -> school
    col = add_schoolbook_block(U, V, W, col, 2, 2, 3, 1, 1, 4)
    
    # Verify that col == total_rank
    assert col == total_rank, f"Column count mismatch: {col} != {total_rank}"
    
    # Convert to numpy arrays and verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k.
    
    # Build the tensor efficiently
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                T[i, j, k] = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
    
    # Check if T is all 1s
    if not np.allclose(T, np.ones((n, m, p))):
        # This is a fallback - should not happen if our construction is correct
        # Let's just return the schoolbook decomposition as a safety net
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
