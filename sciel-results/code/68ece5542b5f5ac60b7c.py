"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-50 decomposition for 2x4x7 matrix multiplication using
    Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    def add_strassen_block(i1, j1, k1, col_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        A block: rows i1,i1+1; cols j1,j1+1
        B block: rows j1,j1+1; cols k1,k1+1
        C block: rows i1,i1+1; cols k1,k1+1
        """
        nonlocal col
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # Contributes to C[i1,k1], C[i2,k2]
        U[i1*m + j1][col_start] = 1
        U[i2*m + j2][col_start] = 1
        V[j1*p + k1][col_start] = 1
        V[j2*p + k2][col_start] = 1
        W[k1*n + i1][col_start] = 1
        W[k2*n + i2][col_start] = 1
        col = col_start + 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # Contributes to C[i2,k1]
        U[i2*m + j1][col_start] = 1
        U[i2*m + j2][col_start] = 1
        V[j1*p + k1][col_start] = 1
        W[k1*n + i2][col_start] = 1
        col = col_start + 1
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        # Contributes to C[i1,k2]
        U[i1*m + j1][col_start] = 1
        V[j1*p + k2][col_start] = 1
        V[j2*p + k2][col_start] = -1
        W[k2*n + i1][col_start] = 1
        col = col_start + 1
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        # Contributes to C[i2,k1]
        U[i2*m + j2][col_start] = 1
        V[j2*p + k1][col_start] = 1
        V[j1*p + k1][col_start] = -1
        W[k1*n + i2][col_start] = 1
        col = col_start + 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # Contributes to C[i1,k2]
        U[i1*m + j1][col_start] = 1
        U[i1*m + j2][col_start] = 1
        V[j2*p + k2][col_start] = 1
        W[k2*n + i1][col_start] = 1
        col = col_start + 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # Contributes to C[i2,k1], C[i2,k2]
        U[i2*m + j1][col_start] = 1
        U[i1*m + j1][col_start] = -1
        V[j1*p + k1][col_start] = 1
        V[j1*p + k2][col_start] = 1
        W[k1*n + i2][col_start] = 1
        W[k2*n + i2][col_start] = 1
        col = col_start + 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # Contributes to C[i1,k1], C[i1,k2]
        U[i1*m + j2][col_start] = 1
        U[i2*m + j2][col_start] = -1
        V[j2*p + k1][col_start] = 1
        V[j2*p + k2][col_start] = 1
        W[k1*n + i1][col_start] = 1
        W[k2*n + i1][col_start] = 1
        col = col_start + 1
        
        return col
    
    def add_schoolbook_2x2x1_block(i1, j1, k1, col_start):
        """
        Add schoolbook decomposition for a 2x2x1 block.
        A block: rows i1,i1+1; cols j1,j1+1
        B block: rows j1,j1+1; col k1
        C block: rows i1,i1+1; col k1
        """
        nonlocal col
        i2 = i1 + 1
        j2 = j1 + 1
        
        # C[i1,k1] += A[i1,j1]*B[j1,k1]
        U[i1*m + j1][col_start] = 1
        V[j1*p + k1][col_start] = 1
        W[k1*n + i1][col_start] = 1
        col = col_start + 1
        
        # C[i1,k1] += A[i1,j2]*B[j2,k1]
        U[i1*m + j2][col_start] = 1
        V[j2*p + k1][col_start] = 1
        W[k1*n + i1][col_start] = 1
        col = col_start + 1
        
        # C[i2,k1] += A[i2,j1]*B[j1,k1]
        U[i2*m + j1][col_start] = 1
        V[j1*p + k1][col_start] = 1
        W[k1*n + i2][col_start] = 1
        col = col_start + 1
        
        # C[i2,k1] += A[i2,j2]*B[j2,k1]
        U[i2*m + j2][col_start] = 1
        V[j2*p + k1][col_start] = 1
        W[k1*n + i2][col_start] = 1
        col = col_start + 1
        
        return col
    
    # Partition m=4 into two groups of 2
    # Partition p=7 into three groups of 2 and one group of 1
    m_blocks = [0, 2]  # start indices for each group of 2
    p_blocks_2 = [0, 2, 4]  # start indices for groups of 2
    p_block_1 = 6  # start index for group of 1
    
    # Add 2x2x2 blocks (Strassen)
    for m_start in m_blocks:
        for p_start in p_blocks_2:
            col = add_strassen_block(0, m_start, p_start, col)
    
    # Add 2x2x1 blocks (schoolbook)
    for m_start in m_blocks:
        col = add_schoolbook_2x2x1_block(0, m_start, p_block_1, col)
    
    assert col == 50, f"Expected col=50, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
