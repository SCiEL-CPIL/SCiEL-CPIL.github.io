import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x3 by 3x9 multiplication as follows:
    # Split A (2x3) into A1 (2x2) and a (2x1)
    # Split B (3x9) into B1 (2x9) and b (1x9)
    # C = A1 * B1 + a * b^T
    # A1 * B1 is 2x2 * 2x9. We split B1 into 4 2x2 blocks and one 2x1 block.
    # Each 2x2 * 2x2 uses Strassen (rank 7). There are 4 such blocks.
    # The 2x2 * 2x1 uses schoolbook (rank 4).
    # So A1*B1 uses 4*7 + 4 = 32 rank terms.
    # a * b^T is 2x1 * 1x9, which is rank 2*1*9 = 18.
    # Total rank = 32 + 18 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50
    
    # Indexing:
    # A is 2x3: A[i, j] where i in 0..1, j in 0..2
    # U index for A[i,j] is i*m + j = i*3 + j
    # B is 3x9: B[j, k] where j in 0..2, k in 0..8
    # V index for B[j,k] is j*p + k = j*9 + k
    # C is 2x9: C[i, k] where i in 0..1, k in 0..8
    # W index for C[i,k] is k*n + i = k*2 + i (trace convention)
    
    col = 0
    
    # Strassen for 2x2 * 2x2:
    # A1 = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For A1 (2x2) corresponding to A[0:2, 0:2]:
    # A00 = A[0,0] -> U index 0*3+0 = 0
    # A01 = A[0,1] -> U index 0*3+1 = 1
    # A10 = A[1,0] -> U index 1*3+0 = 3
    # A11 = A[1,1] -> U index 1*3+1 = 4
    
    # For each 2x2 block of B1 (which is B[0:2, 0:8] split into 4 2x2 blocks):
    # Block t (t=0..3) corresponds to B[0:2, 2*t:2*t+2]
    # B00 = B[0, 2*t] -> V index 0*9 + 2*t = 2*t
    # B01 = B[0, 2*t+1] -> V index 0*9 + 2*t+1 = 2*t+1
    # B10 = B[1, 2*t] -> V index 1*9 + 2*t = 9 + 2*t
    # B11 = B[1, 2*t+1] -> V index 1*9 + 2*t+1 = 9 + 2*t+1
    
    # Output C for block t: C[0:2, 2*t:2*t+2]
    # C00 = C[0, 2*t] -> W index 2*t*2 + 0 = 4*t
    # C01 = C[0, 2*t+1] -> W index (2*t+1)*2 + 0 = 4*t+2
    # C10 = C[1, 2*t] -> W index 2*t*2 + 1 = 4*t+1
    # C11 = C[1, 2*t+1] -> W index (2*t+1)*2 + 1 = 4*t+3
    
    for t in range(4):
        # Strassen block for A1 * B1_block_t
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1
        U[4][col] += 1
        V[2*t][col] += 1
        V[9 + 2*t + 1][col] += 1
        W[4*t][col] += 1      # C00
        W[4*t + 2][col] += 0  # not in M1
        W[4*t + 1][col] += 0
        W[4*t + 3][col] += 1  # C11
        col += 1
        
        # M2 = (A10+A11)B00
        U[3][col] += 1
        U[4][col] += 1
        V[2*t][col] += 1
        W[4*t + 1][col] += 1  # C10
        col += 1
        
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[2*t + 1][col] += 1
        V[9 + 2*t + 1][col] += -1
        W[4*t + 2][col] += 1  # C01
        W[4*t + 3][col] += 1  # C11
        col += 1
        
        # M4 = A11(B10-B00)
        U[4][col] += 1
        V[9 + 2*t][col] += 1
        V[2*t][col] += -1
        W[4*t][col] += 1      # C00
        W[4*t + 1][col] += 1  # C10
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] += 1
        U[1][col] += 1
        V[9 + 2*t + 1][col] += 1
        W[4*t][col] += -1     # C00
        W[4*t + 2][col] += 1  # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[3][col] += 1
        U[0][col] += -1
        V[2*t][col] += 1
        V[2*t + 1][col] += 1
        W[4*t + 3][col] += 1  # C11
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1
        U[4][col] += -1
        V[9 + 2*t][col] += 1
        V[9 + 2*t + 1][col] += 1
        W[4*t][col] += 1      # C00
        col += 1
    
    # Now handle the last column of B1: B[0:2, 8]
    # This is 2x2 * 2x1. Schoolbook: 4 multiplications.
    # A1 = [[A00, A01], [A10, A11]]
    # B_last = [B0, B1]^T where B0 = B[0,8], B1 = B[1,8]
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # V indices: B[0,8] -> 0*9+8 = 8, B[1,8] -> 1*9+8 = 17
    # W indices: C[0,8] -> 8*2+0 = 16, C[1,8] -> 8*2+1 = 17
    
    # C0 = A00*B0 + A01*B1
    U[0][col] += 1
    V[8][col] += 1
    W[16][col] += 1
    col += 1
    
    U[1][col] += 1
    V[17][col] += 1
    W[16][col] += 1
    col += 1
    
    # C1 = A10*B0 + A11*B1
    U[3][col] += 1
    V[8][col] += 1
    W[17][col] += 1
    col += 1
    
    U[4][col] += 1
    V[17][col] += 1
    W[17][col] += 1
    col += 1
    
    # Now handle a * b^T where a = A[0:2, 2] (2x1) and b = B[2, 0:9] (1x9)
    # a = [a0, a1]^T where a0 = A[0,2] -> U index 0*3+2 = 2, a1 = A[1,2] -> U index 1*3+2 = 5
    # b = [b0, ..., b8] where bk = B[2,k] -> V index 2*9+k = 18+k
    # C[i,k] += a_i * b_k
    # W index for C[i,k] is k*2 + i
    
    for k in range(9):
        # C[0,k] += a0 * b_k
        U[2][col] += 1
        V[18 + k][col] += 1
        W[k * 2 + 0][col] += 1
        col += 1
        
        # C[1,k] += a1 * b_k
        U[5][col] += 1
        V[18 + k][col] += 1
        W[k * 2 + 1][col] += 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
