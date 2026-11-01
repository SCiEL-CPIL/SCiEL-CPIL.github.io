"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # We'll use Strassen for the 2x2x2 part (columns 0,1 of B) 
    # and schoolbook for columns 2,3,4.
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A10 - A00) * B00
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 + A01) * B00
    
    # Output:
    # C00 = M5 + M4 - M2 + M1
    # C01 = M1 + M7
    # C10 = M3 + M2
    # C11 = M5 + M3 - M1 - M6
    
    # U rows: i*m + j where i is row of A (0,1), j is column of A (0,1)
    # V rows: j*p + k where j is row of B (0,1), k is column of B (0,1,2,3,4)
    # W rows: k*n + i where k is column of C (0..4), i is row of C (0,1)
    
    # For Strassen part (columns 0,1 of B):
    # U entries for Strassen:
    # M1: A00 * (B01 - B11) -> U[0][c1] = 1 (A00), V[0*p+1][c1] = 1 (B01), V[1*p+1][c1] = -1 (B11)
    #     But wait, V is for B entries. B has shape (m, p) = (2, 5).
    #     V rows are indexed by j*p + k, where j is row of B (0,1) and k is column of B (0..4).
    
    # Let me re-think the tensor decomposition.
    # The matrix multiplication tensor T has T[i,j,k] = 1 if C[i,k] = sum A[i,j] B[j,k]
    # A rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # So for each term r, U[i,j,r] picks A[i,j], V[j,k,r] picks B[j,k], W[k,i,r] contributes to C[k,i]... 
    # Wait, the convention says W[k*n+i, r]. Let me re-read.
    # "W[k*n+i,r]" - so W is indexed by (output_col, output_row)? Or is it W indexed by k*n+i?
    # Actually, looking at the seed program: W[k * n + i][column] = 1
    # And C[i,k] = A[i,:] * B[:,k]
    # So the output C has shape (n, p) = (2, 5).
    # W[k*n+i] suggests k is column index of C, i is row index of C.
    # But C has shape (n, p) = (2, 5), so row index goes 0..1, column index goes 0..4.
    # k*n + i with n=2: if k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3; etc.
    # So W row index = k * n + i, where k is column of C (0..p-1=0..4), i is row of C (0..n-1=0..1).
    
    # OK so the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i, j, k
    
    # Let me build the Strassen part for columns k=0,1 of C (i.e., B columns 0,1):
    
    rank_strassen = 7
    rank_schoolbook = (p - 2) * m * n  # 3 columns * 2 * 2 = 12
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * total_rank for _ in range(m * p)]  # 10 x rank
    W = [[0] * total_rank for _ in range(p * n)]  # 10 x rank
    
    # Strassen terms (columns 0..6)
    # M1 = A00 * (B01 - B11)
    r = 0
    U[0*2+0][r] = 1  # A00
    V[0*5+1][r] = 1  # B01
    V[1*5+1][r] = -1 # B11
    # C00 += M1, C01 += M1
    W[0*2+0][r] = 1  # C00 (k=0, i=0)
    W[1*2+0][r] = 1  # C01 (k=1, i=0)
    
    # M2 = (A00 + A10) * B11
    r = 1
    U[0*2+0][r] = 1  # A00
    U[1*2+0][r] = 1  # A10
    V[1*5+1][r] = 1  # B11
    # C00 -= M2, C10 += M2
    W[0*2+0][r] = -1 # C00
    W[0*2+1][r] = 1  # C10 (k=0, i=1)
    
    # M3 = (A10 - A00) * B00
    r = 2
    U[1*2+0][r] = 1  # A10
    U[0*2+0][r] = -1 # A00
    V[0*5+0][r] = 1  # B00
    # C10 += M3, C11 += M3
    W[0*2+1][r] = 1  # C10
    W[1*2+1][r] = 1  # C11 (k=1, i=1)
    
    # M4 = A11 * (B00 - B10)
    r = 3
    U[1*2+1][r] = 1  # A11
    V[0*5+0][r] = 1  # B00
    V[1*5+0][r] = -1 # B10
    # C00 += M4, C11 -= M4
    W[0*2+0][r] = 1  # C00
    W[1*2+1][r] = -1 # C11
    
    # M5 = (A00 + A11) * (B00 + B11)
    r = 4
    U[0*2+0][r] = 1  # A00
    U[1*2+1][r] = 1  # A11
    V[0*5+0][r] = 1  # B00
    V[1*5+1][r] = 1  # B11
    # C00 += M5, C11 += M5
    W[0*2+0][r] = 1  # C00
    W[1*2+1][r] = 1  # C11
    
    # M6 = (A01 - A11) * (B10 + B11)
    r = 5
    U[0*2+1][r] = 1  # A01
    U[1*2+1][r] = -1 # A11
    V[1*5+0][r] = 1  # B10
    V[1*5+1][r] = 1  # B11
    # C11 -= M6
    W[1*2+1][r] = -1 # C11
    
    # M7 = (A00 + A01) * B00
    r = 6
    U[0*2+0][r] = 1  # A00
    U[0*2+1][r] = 1  # A01
    V[0*5+0][r] = 1  # B00
    # C01 += M7
    W[1*2+0][r] = 1  # C01
    
    # Schoolbook for columns k=2,3,4
    r = 7
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
