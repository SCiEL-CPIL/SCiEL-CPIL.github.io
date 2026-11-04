"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Split into two 2x2x5 sub-problems.
    Each 2x2x5 splits into two 2x2x2 (Strassen, rank 7) + one 2x2x1 (schoolbook, rank 4).
    Total rank: 2 * (7 + 7 + 4) = 36.
    
    U: shape (n*m, rank) = (8, 36)
    V: shape (m*p, rank) = (20, 36)  
    W: shape (p*n, rank) = (10, 36)
    
    Indexing:
    U[i*m+j, c] where i in [0,2), j in [0,4)
    V[j*p+k, c] where j in [0,4), k in [0,5)
    W[k*n+i, c] where k in [0,5), i in [0,2)
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # We'll build the decomposition column by column.
    # Each "block" contributes a set of columns.
    
    # Helper: add a Strassen 2x2x2 block
    # A is 2x2 (rows i1,i2; cols j1,j2), B is 2x2 (rows j1,j2; cols k1,k2), C is 2x2 (rows i1,i2; cols k1,k2)
    # Strassen's 7 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, each multiplication M_t corresponds to one column.
    # U entries are the coefficients of A elements in M_t
    # V entries are the coefficients of B elements in M_t
    # W entries are the coefficients of C elements that M_t contributes to
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] = C[i1:i2+1, k1:k2+1]"""
        # A indices: (i1,j1)=A11, (i1,j2)=A12, (i2,j1)=A21, (i2,j2)=A22
        # B indices: (j1,k1)=B11, (j1,k2)=B12, (j2,k1)=B21, (j2,k2)=B22
        # C indices: (i1,k1)=C11, (i1,k2)=C12, (i2,k1)=C21, (i2,k2)=C22
        
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (A11+A22)(B11+B22)
        col = col_offset + 0
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        
        # M2 = (A21+A22)B11
        col = col_offset + 1
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        
        # M3 = A11(B12-B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1
        V[b22][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        
        # M4 = A22(B21-B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        
        # M5 = (A11+A12)B22
        col = col_offset + 4
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        
        # M6 = (A21-A11)(B11+B12)
        col = col_offset + 5
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        
        # M7 = (A12-A22)(B21+B22)
        col = col_offset + 6
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
    
    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        """Add schoolbook 2x2x1 block for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1] = C[i1:i2+1, k1]"""
        # 4 terms: A11*B11, A12*B21, A21*B11, A22*B21
        # C11 = A11*B11 + A12*B21
        # C21 = A21*B11 + A22*B21
        
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        # Term 1: A11 * B11 -> contributes to C11
        col = col_offset + 0
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        
        # Term 2: A12 * B21 -> contributes to C11
        col = col_offset + 1
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        
        # Term 3: A21 * B11 -> contributes to C21
        col = col_offset + 2
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        
        # Term 4: A22 * B21 -> contributes to C21
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1
    
    # Now build the full decomposition.
    # A is 2x4, B is 4x5, C is 2x5.
    # Split B's rows into two groups of 2: rows 0,1 and rows 2,3.
    # This gives two sub-problems:
    #   Sub1: A[0:2, 0:2] * B[0:2, 0:5] = C1[0:2, 0:5]
    #   Sub2: A[0:2, 2:4] * B[2:4, 0:5] = C2[0:2, 0:5]
    #   C = C1 + C2
    
    # For each 2x2x5, split columns into 2+2+1:
    #   2x2x2 (cols 0,1): Strassen rank 7
    #   2x2x2 (cols 2,3): Strassen rank 7
    #   2x2x1 (col 4): schoolbook rank 4
    # Total per 2x2x5: 18, total: 36.
    
    col = 0
    
    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    # 2x2x2 for cols 0,1
    add_strassen_222(0, 1, 0, 1, 0, 1, col)
    col += 7
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 0, 1, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 0, 1, 4, col)
    col += 4
    
    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-4
    # 2x2x2 for cols 0,1
    add_strassen_222(0, 1, 2, 3, 0, 1, col)
    col += 7
    # 2x2x2 for cols 2,3
    add_strassen_222(0, 1, 2, 3, 2, 3, col)
    col += 7
    # 2x2x1 for col 4
    add_schoolbook_221(0, 1, 2, 3, 4, col)
    col += 4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
