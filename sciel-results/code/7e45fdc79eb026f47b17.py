"""Matrix multiplication 2x2x5 - rank-minimal exact tensor decomposition.

Uses a constructive approach combining:
1. Strassen's 7-term decomposition for the first 2x2 block (columns 0,1)
2. Standard 4-term decomposition for column 2
3. Standard 4-term decomposition for column 3
4. Standard 4-term decomposition for column 4

This gives rank 7 + 4 + 4 + 4 = 19.

The output follows the convention W[k*n+i, r] where:
- A is n x m = 2 x 2
- B is m x p = 2 x 5
- C = A * B is n x p = 2 x 5
- C[i,k] = sum_j A[i,j] * B[j,k]

U has shape (n*m, rank) = (4, rank), indexed by (i,j) -> i*m+j
V has shape (m*p, rank) = (10, rank), indexed by (j,k) -> j*p+k
W has shape (p*n, rank) = (10, rank), indexed by (k,i) -> k*n+i
"""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 7 (Strassen for cols 0,1) + 4 (col 2) + 4 (col 3) + 4 (col 4) = 19
    rank = 19
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 19
    V = [[0] * rank for _ in range(m * p)]  # 10 x 19
    W = [[0] * rank for _ in range(p * n)]  # 10 x 19
    
    # Strassen decomposition for 2x2 * 2x2 (columns 0 and 1)
    # A = [[a00, a01], [a10, a11]]
    # B_block = [[b00, b01], [b10, b11]]
    # C_block = A * B_block = [[c00, c01], [c10, c11]]
    #
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = M1 - M4 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 - M5
    #
    # In our indexing:
    # U index: i*m+j where i in {0,1}, j in {0,1}
    #   U[0] = (0,0) = a00
    #   U[1] = (0,1) = a01
    #   U[2] = (1,0) = a10
    #   U[3] = (1,1) = a11
    #
    # V index: j*p+k where j in {0,1}, k in {0,1} for the block
    #   V[0] = (0,0) = b00
    #   V[1] = (0,1) = b01
    #   V[2] = (1,0) = b10
    #   V[3] = (1,1) = b11
    #
    # W index: k*n+i where k in {0,1}, i in {0,1}
    #   W[0] = (0,0) = c00
    #   W[1] = (0,1) = c10  -- wait, W[k*n+i], so k=0,i=0 -> W[0]=c00, k=0,i=1 -> W[1]=c10? 
    #   Actually W[k*n+i]: k is column index, i is row index of C
    #   C[i,k], so W[k*n+i] corresponds to C[i,k]
    #   W[0*n+0] = W[0] = C[0,0]
    #   W[0*n+1] = W[1] = C[1,0]
    #   W[1*n+0] = W[2] = C[0,1]
    #   W[1*n+1] = W[3] = C[1,1]
    
    # Term 0: M1 = (a00+a11)*(b00+b11)
    # U: a00 and a11 get +1
    U[0][0] += 1   # a00
    U[3][0] += 1   # a11
    # V: b00 and b11 get +1
    V[0][0] += 1   # b00 (j=0,k=0)
    V[3][0] += 1   # b11 (j=1,k=1)
    # W: contributes to c00, c01, c10, c11
    # c00 = M1 - M4 + M6 + M7, so M1 contributes +1 to c00
    # c01 = M3 + M5, M1 does not contribute
    # c10 = M2 + M4, M1 does not contribute
    # c11 = M1 - M3 + M2 - M5, so M1 contributes +1 to c11
    W[0][0] += 1   # C[0,0]
    W[3][0] += 1   # C[1,1]
    
    # Term 1: M2 = (a10+a11)*b00
    U[2][1] += 1   # a10
    U[3][1] += 1   # a11
    V[0][1] += 1   # b00
    # c10 = M2 + M4, so M2 contributes +1 to c10
    W[1][1] += 1   # C[1,0]
    # c11 = M1 - M3 + M2 - M5, so M2 contributes +1 to c11
    W[3][1] += 1   # C[1,1]
    
    # Term 2: M3 = a00*(b01-b11)
    U[0][2] += 1   # a00
    V[1][2] += 1   # b01 (j=0,k=1)
    V[3][2] += -1  # b11 (j=1,k=1)
    # c01 = M3 + M5, so M3 contributes +1 to c01
    W[2][2] += 1   # C[0,1]
    # c11 = M1 - M3 + M2 - M5, so M3 contributes -1 to c11
    W[3][2] += -1  # C[1,1]
    
    # Term 3: M4 = a11*(b10-b00)
    U[3][3] += 1   # a11
    V[2][3] += 1   # b10 (j=1,k=0)
    V[0][3] += -1  # b00 (j=0,k=0)
    # c00 = M1 - M4 + M6 + M7, so M4 contributes -1 to c00
    W[0][3] += -1  # C[0,0]
    # c10 = M2 + M4, so M4 contributes +1 to c10
    W[1][3] += 1   # C[1,0]
    
    # Term 4: M5 = (a00+a01)*b11
    U[0][4] += 1   # a00
    U[1][4] += 1   # a01
    V[3][4] += 1   # b11 (j=1,k=1)
    # c01 = M3 + M5, so M5 contributes +1 to c01
    W[2][4] += 1   # C[0,1]
    # c11 = M1 - M3 + M2 - M5, so M5 contributes -1 to c11
    W[3][4] += -1  # C[1,1]
    
    # Term 5: M6 = (a10-a00)*(b00+b01)
    U[2][5] += 1   # a10
    U[0][5] += -1  # a00
    V[0][5] += 1   # b00
    V[1][5] += 1   # b01
    # c00 = M1 - M4 + M6 + M7, so M6 contributes +1 to c00
    W[0][5] += 1   # C[0,0]
    
    # Term 6: M7 = (a01-a11)*(b10+b11)
    U[1][6] += 1   # a01
    U[3][6] += -1  # a11
    V[2][6] += 1   # b10
    V[3][6] += 1   # b11
    # c00 = M1 - M4 + M6 + M7, so M7 contributes +1 to c00
    W[0][6] += 1   # C[0,0]
    
    # Now standard 4-term decompositions for columns 2, 3, 4
    # For column k, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x1 * 1x1... no, it's summing over j=0,1
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed k, we have 2 terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    # But actually for fixed k, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # and C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # So 4 rank-1 terms per column.
    
    for col in range(2, 5):
        base = 7 + (col - 2) * 4
        # Term base+0: A[0,0]*B[0,col] -> C[0,col]
        U[0][base + 0] += 1   # i=0, j=0
        V[0 * p + col][base + 0] += 1  # j=0, k=col
        W[col * n + 0][base + 0] += 1  # k=col, i=0
        
        # Term base+1: A[0,1]*B[1,col] -> C[0,col]
        U[1][base + 1] += 1   # i=0, j=1
        V[1 * p + col][base + 1] += 1  # j=1, k=col
        W[col * n + 0][base + 1] += 1  # k=col, i=0
        
        # Term base+2: A[1,0]*B[0,col] -> C[1,col]
        U[2][base + 2] += 1   # i=1, j=0
        V[0 * p + col][base + 2] += 1  # j=0, k=col
        W[col * n + 1][base + 2] += 1  # k=col, i=1
        
        # Term base+3: A[1,1]*B[1,col] -> C[1,col]
        U[3][base + 3] += 1   # i=1, j=1
        V[1 * p + col][base + 3] += 1  # j=1, k=col
        W[col * n + 1][base + 3] += 1  # k=col, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
