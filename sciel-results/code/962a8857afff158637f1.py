"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition.

Uses Strassen's algorithm on 2x2x2 subproblems to reduce rank below schoolbook.
Schoolbook: 3*4*8 = 96.
Strategy:
  - Split A (3x4) into top (2x4) and bottom (1x4).
  - Split B (4x8) into 4 blocks of (4x2) each.
  - For each block j (j=0..3):
      * Top part: (2x4) * (4x2) -> (2x2). Split into two (2x2)*(2x2) Strassen blocks.
        Each Strassen block: rank 7. Two per j: rank 14.
      * Bottom part: (1x4) * (4x2) -> (1x2). Schoolbook: rank 4.
      * Total per j: 14 + 4 = 18.
  - Total rank: 4 * 18 = 72.
"""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 72  # 4 blocks * (2*7 + 4) = 4 * 18 = 72
    
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*3 = 24 rows
    
    col = 0
    
    # Strassen 2x2x2 decomposition:
    # Given A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    # C = A*B, C[i][k] = sum_j A[i][j]*B[j][k]
    # Strassen terms:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1, base_col):
        """Add Strassen decomposition for A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1]
           where i0,i1 are row indices in A, j0,j1 col indices in A / row indices in B,
           k0,k1 col indices in B.
           Returns next column index.
        """
        # A indices: rows i0, i1; cols j0, j1
        # B indices: rows j0, j1; cols k0, k1
        # C indices: rows i0, i1; cols k0, k1
        
        a11 = (i0, j0)  # A[i0][j0]
        a12 = (i0, j1)  # A[i0][j1]
        a21 = (i1, j0)  # A[i1][j0]
        a22 = (i1, j1)  # A[i1][j1]
        
        b11 = (j0, k0)  # B[j0][k0]
        b12 = (j0, k1)  # B[j0][k1]
        b21 = (j1, k0)  # B[j1][k0]
        b22 = (j1, k1)  # B[j1][k1]
        
        c11 = (i0, k0)  # C[i0][k0]
        c12 = (i0, k1)  # C[i0][k1]
        c21 = (i1, k0)  # C[i1][k0]
        c22 = (i1, k1)  # C[i1][k1]
        
        def add_term(u_row, u_coeff, v_row, v_coeff, w_terms):
            """w_terms is list of (w_row, w_coeff)"""
            nonlocal col
            U[u_row][col] = u_coeff
            V[v_row][col] = v_coeff
            for w_row, w_coeff in w_terms:
                W[w_row][col] = w_coeff
            col += 1
        
        # M1 = (a11 + a22) * (b11 + b22)
        # U: a11 + a22 => rows i0,j0 and i1,j1 both get +1
        # V: b11 + b22 => rows j0,k0 and j1,k1 both get +1
        # W: C11 +1, C22 +1
        add_term(a11[0]*m + a11[1], 1, b11[0]*p + b11[1], 1, [(c11[0]*n + c11[1], 1), (c22[0]*n + c22[1], 1)])
        # Wait, we need to add two U entries for a11+a22. Let me restructure.
        
        # Actually, each Strassen term is a single rank-1 tensor.
        # M1: U has +1 at (i0,j0) and +1 at (i1,j1); V has +1 at (j0,k0) and +1 at (j1,k1); W has +1 at (i0,k0) and +1 at (i1,k1)
        # But this isn't a single rank-1 term - it's a sum of 4 rank-1 terms if done naively.
        # 
        # Wait, no. In Strassen, each "term" M1 corresponds to ONE rank-1 decomposition where:
        # U vector has entries at the positions corresponding to (a11+a22), i.e., U[(i0,j0)]=1, U[(i1,j1)]=1
        # V vector has entries at (b11+b22), i.e., V[(j0,k0)]=1, V[(j1,k1)]=1
        # W vector has entries at (c11, c22) with appropriate signs
        #
        # So each Strassen M is ONE rank-1 term with multiple non-zero entries in U, V, W.
        
        col -= 1  # Undo the previous incorrect addition
        
        # M1: (a11+a22)*(b11+b22) -> contributes +1 to C11, +1 to C22
        U[a11[0]*m + a11[1]][col] = 1
        U[a22[0]*m + a22[1]][col] = 1
        V[b11[0]*p + b11[1]][col] = 1
        V[b22[0]*p + b22[1]][col] = 1
        W[c11[0]*n + c11[1]][col] = 1
        W[c22[0]*n + c22[1]][col] = 1
        col += 1
        
        # M2: (a21+a22)*b11 -> contributes +1 to C21, -1 to C22
        U[a21[0]*m + a21[1]][col] = 1
        U[a22[0]*m + a22[1]][col] = 1
        V[b11[0]*p + b11[1]][col] = 1
        W[c21[0]*n + c21[1]][col] = 1
        W[c22[0]*n + c22[1]][col] = -1
        col += 1
        
        # M3: a11*(b12-b22) -> contributes +1 to C12, +1 to C22
        U[a11[0]*m + a11[1]][col] = 1
        V[b12[0]*p + b12[1]][col] = 1
        V[b22[0]*p + b22[1]][col] = -1
        W[c12[0]*n + c12[1]][col] = 1
        W[c22[0]*n + c22[1]][col] = 1
        col += 1
        
        # M4: a22*(b21-b11) -> contributes +1 to C11, +1 to C21
        U[a22[0]*m + a22[1]][col] = 1
        V[b21[0]*p + b21[1]][col] = 1
        V[b11[0]*p + b11[1]][col] = -1
        W[c11[0]*n + c11[1]][col] = 1
        W[c21[0]*n + c21[1]][col] = 1
        col += 1
        
        # M5: (a11+a12)*b22 -> contributes -1 to C11, +1 to C12
        U[a11[0]*m + a11[1]][col] = 1
        U[a12[0]*m + a12[1]][col] = 1
        V[b22[0]*p + b22[1]][col] = 1
        W[c11[0]*n + c11[1]][col] = -1
        W[c12[0]*n + c12[1]][col] = 1
        col += 1
        
        # M6: (a21-a11)*(b11+b12) -> contributes +1 to C22
        U[a21[0]*m + a21[1]][col] = 1
        U[a11[0]*m + a11[1]][col] = -1
        V[b11[0]*p + b11[1]][col] = 1
        V[b12[0]*p + b12[1]][col] = 1
        W[c22[0]*n + c22[1]][col] = 1
        col += 1
        
        # M7: (a12-a22)*(b21+b22) -> contributes +1 to C11
        U[a12[0]*m + a12[1]][col] = 1
        U[a22[0]*m + a22[1]][col] = -1
        V[b21[0]*p + b21[1]][col] = 1
        V[b22[0]*p + b22[1]][col] = 1
        W[c11[0]*n + c11[1]][col] = 1
        col += 1
        
        return col
    
    # For each block j (0..3):
    for j in range(4):
        # B block j uses rows 0..3, cols 2j..2j+1
        k0 = 2 * j
        k1 = 2 * j + 1
        
        # Bottom part: A row 2 (1x4) * B block j (4x2) -> C row 2, cols 2j..2j+1 (1x2)
        # Schoolbook: 4 terms
        for jj in range(4):  # j index in A cols / B rows
            for kk in range(2):  # k index in B cols
                U[2 * m + jj][col] = 1  # A[2][jj]
                V[jj * p + k0 + kk][col] = 1  # B[jj][k0+kk]
                W[(k0 + kk) * n + 2][col] = 1  # C[2][k0+kk]
                col += 1
        
        # Top part: A rows 0..1 (2x4) * B block j (4x2) -> C rows 0..1, cols 2j..2j+1 (2x2)
        # Split A cols into two pairs: (0,1) and (2,3)
        # Split B rows into two pairs: (0,1) and (2,3)
        
        # Strassen block 1: A[0:2, 0:2] * B[0:2, 2j:2j+2]
        add_strassen_block(0, 1, 0, 1, k0, k1, col)
        
        # Strassen block 2: A[0:2, 2:4] * B[2:4, 2j:2j+2]
        add_strassen_block(0, 1, 2, 3, k0, k1, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
