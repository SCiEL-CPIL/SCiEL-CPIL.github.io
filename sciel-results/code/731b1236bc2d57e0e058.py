"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # 8 blocks of 2x2x2 Strassen (rank 7) + 8 blocks of 1x2x2 schoolbook (rank 4) = 88
    total_rank = 8 * 7 + 8 * 4  # 56 + 32 = 88
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        """
        Strassen 2x2x2 block.
        A submatrix: rows i0,i0+1; cols j0,j0+1
        B submatrix: rows j0,j0+1; cols k0,k0+1
        C submatrix: rows k0,k0+1; cols i0,i0+1
        
        W row for C[k, i] is at index k*n + i
        U row for A[i, j] is at index i*m + j
        V row for B[j, k] is at index j*p + k
        """
        nonlocal col
        
        # m1 = (a11+a22)*(b11+b22)
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1        # c11 += m1
        W[(k0+1)*n + (i0+1)][col] = 1  # c22 += m1
        col += 1
        
        # m2 = (a21+a22)*b11
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[(k0+1)*n + i0][col] = 1      # c21 += m2
        W[(k0+1)*n + (i0+1)][col] = -1  # c22 -= m2
        col += 1
        
        # m3 = a11*(b12-b22)
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1      # c12 += m3
        W[(k0+1)*n + (i0+1)][col] = 1  # c22 += m3
        col += 1
        
        # m4 = a22*(b21-b11)
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = -1         # c11 -= m4
        W[(k0+1)*n + i0][col] = 1      # c21 += m4
        col += 1
        
        # m5 = (a11+a12)*b22
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1          # c11 += m5
        W[(k0+1)*n + i0][col] = 1      # c12 += m5
        col += 1
        
        # m6 = (a21-a11)*(b11+b12)
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1          # c11 += m6
        col += 1
        
        # m7 = (a12-a22)*(b21+b22)
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1  # c22 += m7
        col += 1
    
    def add_schoolbook_1x2x2(i0, j0, k0):
        """Schoolbook 1x2x2: one input row i0, 2 middle (j0,j0+1), 2 output (k0,k0+1)."""
        nonlocal col
        for j in range(2):
            for k in range(2):
                U[i0*m + (j0+j)][col] = 1
                V[(j0+j)*p + (k0+k)][col] = 1
                W[(k0+k)*n + i0][col] = 1
                col += 1
    
    # 2x2x2 Strassen blocks: input rows 0-1, middle groups (0-1, 2-3), output groups (0-1, 2-3, 4-5, 6-7)
    for j0 in range(0, m, 2):      # j0 = 0, 2
        for k0 in range(0, p, 2):  # k0 = 0, 2, 4, 6
            add_strassen_2x2x2(0, j0, k0)
    
    # 1x2x2 schoolbook blocks: input row 2, middle groups (0-1, 2-3), output groups (0-1, 2-3, 4-5, 6-7)
    for j0 in range(0, m, 2):      # j0 = 0, 2
        for k0 in range(0, p, 2):  # k0 = 0, 2, 4, 6
            add_schoolbook_1x2x2(2, j0, k0)
    
    assert col == total_rank, f"Column count {col} != expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
