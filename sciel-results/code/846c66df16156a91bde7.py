"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition with Strassen on 2x2x2 sub-blocks
    # Partition j (cols of A / rows of B) into J1={0,1}, J2={2,3}, J3={4}
    # Partition k (cols of B / cols of C) into K1={0,1}, K2={2,3}, K3={4}
    # i (rows of A / cols of C) = {0,1}
    #
    # Blocks and their ranks:
    # (i,J1,K1): 2x2x2 -> Strassen 7
    # (i,J1,K2): 2x2x2 -> Strassen 7
    # (i,J1,K3): 2x2x1 -> schoolbook 4
    # (i,J2,K1): 2x2x2 -> Strassen 7
    # (i,J2,K2): 2x2x2 -> Strassen 7
    # (i,J2,K3): 2x2x1 -> schoolbook 4
    # (i,J3,K1): 2x1x2 -> schoolbook 4
    # (i,J3,K2): 2x1x2 -> schoolbook 4
    # (i,J3,K3): 2x1x1 -> schoolbook 2
    # Total rank = 7+7+4+7+7+4+4+4+2 = 46
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 46
    V = [[0] * rank for _ in range(m * p)]  # 25 x 46
    W = [[0] * rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For matrices A(2x2), B(2x2), C(2x2):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_block(i_offset, j_indices, k_indices, col_start):
        """Add Strassen decomposition for a 2x2x2 block.
        i_offset: starting row index for i (0 or 1, but here always 0 since n=2)
        j_indices: list of 2 j indices
        k_indices: list of 2 k indices
        col_start: starting column
        Returns new col_start
        """
        j1, j2 = j_indices
        k1, k2 = k_indices
        # i indices are 0 and 1 (the two rows)
        
        # M1 = (A[0,j1]+A[1,j2]) * (B[j1,k1]+B[j2,k2])
        # A part: row 0 col j1 = 1, row 1 col j2 = 1
        # B part: row j1 col k1 = 1, row j2 col k2 = 1
        U[0 * m + j1][col_start] = 1
        U[1 * m + j2][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        # C11 += M1, C22 += M1
        W[k1 * n + 0][col_start] = 1
        W[k2 * n + 1][col_start] = 1
        col_start += 1
        
        # M2 = (A[1,j1]+A[1,j2]) * B[j1,k1]
        U[1 * m + j1][col_start] = 1
        U[1 * m + j2][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        # C21 += M2
        W[k1 * n + 1][col_start] = 1
        col_start += 1
        
        # M3 = A[0,j1] * (B[j1,k2]-B[j2,k2])
        U[0 * m + j1][col_start] = 1
        V[j1 * p + k2][col_start] = 1
        V[j2 * p + k2][col_start] = -1
        # C12 += M3
        W[k2 * n + 0][col_start] = 1
        col_start += 1
        
        # M4 = A[1,j2] * (B[j2,k1]-B[j1,k1])
        U[1 * m + j2][col_start] = 1
        V[j2 * p + k1][col_start] = 1
        V[j1 * p + k1][col_start] = -1
        # C11 += M4, C21 += M4
        W[k1 * n + 0][col_start] = 1
        W[k1 * n + 1][col_start] = 1
        col_start += 1
        
        # M5 = (A[0,j1]+A[0,j2]) * B[j2,k2]
        U[0 * m + j1][col_start] = 1
        U[0 * m + j2][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        # C11 -= M5, C12 += M5
        W[k1 * n + 0][col_start] = -1
        W[k2 * n + 0][col_start] = 1
        col_start += 1
        
        # M6 = (A[1,j1]-A[0,j1]) * (B[j1,k1]+B[j1,k2])
        U[1 * m + j1][col_start] = 1
        U[0 * m + j1][col_start] = -1
        V[j1 * p + k1][col_start] = 1
        V[j1 * p + k2][col_start] = 1
        # C22 += M6
        W[k2 * n + 1][col_start] = 1
        col_start += 1
        
        # M7 = (A[0,j2]-A[1,j2]) * (B[j2,k1]+B[j2,k2])
        U[0 * m + j2][col_start] = 1
        U[1 * m + j2][col_start] = -1
        V[j2 * p + k1][col_start] = 1
        V[j2 * p + k2][col_start] = 1
        # C11 += M7
        W[k1 * n + 0][col_start] = 1
        col_start += 1
        
        return col_start
    
    def add_schoolbook_2x2x1(i_indices, j_indices, k_indices, col_start):
        """Add schoolbook for 2x2x1 block (2 i's, 2 j's, 1 k)."""
        i1, i2 = i_indices
        j1, j2 = j_indices
        k1 = k_indices[0]
        # C[i,k] += sum_j A[i,j]*B[j,k]
        # For each i and j:
        for i in [i1, i2]:
            for j in [j1, j2]:
                U[i * m + j][col_start] = 1
                V[j * p + k1][col_start] = 1
                W[k1 * n + i][col_start] = 1
                col_start += 1
        return col_start
    
    def add_schoolbook_2x1x2(i_indices, j_indices, k_indices, col_start):
        """Add schoolbook for 2x1x2 block (2 i's, 1 j, 2 k's)."""
        i1, i2 = i_indices
        j1 = j_indices[0]
        k1, k2 = k_indices
        for i in [i1, i2]:
            for k in [k1, k2]:
                U[i * m + j1][col_start] = 1
                V[j1 * p + k][col_start] = 1
                W[k * n + i][col_start] = 1
                col_start += 1
        return col_start
    
    def add_schoolbook_2x1x1(i_indices, j_indices, k_indices, col_start):
        """Add schoolbook for 2x1x1 block (2 i's, 1 j, 1 k)."""
        i1, i2 = i_indices
        j1 = j_indices[0]
        k1 = k_indices[0]
        for i in [i1, i2]:
            U[i * m + j1][col_start] = 1
            V[j1 * p + k1][col_start] = 1
            W[k1 * n + i][col_start] = 1
            col_start += 1
        return col_start
    
    # Define index groups
    i_all = [0, 1]
    J1 = [0, 1]
    J2 = [2, 3]
    J3 = [4]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4]
    
    # 2x2x2 Strassen blocks
    col = add_strassen_block(0, J1, K1, col)  # 7
    col = add_strassen_block(0, J1, K2, col)  # 7
    col = add_strassen_block(0, J2, K1, col)  # 7
    col = add_strassen_block(0, J2, K2, col)  # 7
    
    # 2x2x1 schoolbook blocks
    col = add_schoolbook_2x2x1(i_all, J1, K3, col)  # 4
    col = add_schoolbook_2x2x1(i_all, J2, K3, col)  # 4
    
    # 2x1x2 schoolbook blocks
    col = add_schoolbook_2x1x2(i_all, J3, K1, col)  # 4
    col = add_schoolbook_2x1x2(i_all, J3, K2, col)  # 4
    
    # 2x1x1 schoolbook block
    col = add_schoolbook_2x1x1(i_all, J3, K3, col)  # 2
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
