"""Rewritten program for 3x3*3x4 matrix multiplication tensor decomposition.

Strategy: Use block decomposition to partition the multiplication into:
- A[0,0] * B[0,:] -> C[0,:]: 1x3 scalar-matrix product, rank 4
- A[0,1:3] * B[1:3,:] -> C[0,1:3]: 1x2 * 2x3, rank 6 (schoolbook)
- A[1:3,0] * B[0,:] -> C[1:3,0]: 2x1 * 1x3, rank 6 (schoolbook)
- A[1:3,1:3] * B[1:3,1:3] -> C[1:3,1:3]: 2x2 * 2x2, rank 7 (Strassen)

Total rank = 4 + 6 + 6 + 7 = 23

Indexing conventions:
- U has shape (n*m, rank) = (9, rank), indexed as U[i*m + j] for A[i,j]
- V has shape (m*p, rank) = (12, rank), indexed as V[j*p + k] for B[j,k]
- W has shape (p*n, rank) = (12, rank), indexed as W[k*n + i] for C[i,k]
"""

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Total rank
    rank = 23
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]   # 9 x 23
    V = [[0] * rank for _ in range(m * p)]   # 12 x 23
    W = [[0] * rank for _ in range(p * n)]   # 12 x 23
    
    col = 0
    
    # Block 1: A[0,0] * B[0,:] -> C[0,:]
    # A[0,0] is at U[0*3+0] = U[0]
    # B[0,k] is at V[0*4+k] = V[k] for k=0,1,2,3
    # C[0,k] is at W[k*3+0] = W[3k] for k=0,1,2,3
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
    
    # Block 2: A[0,1:3] * B[1:3,:] -> C[0,1:3]
    # A[0,1] at U[0*3+1] = U[1], A[0,2] at U[0*3+2] = U[2]
    # B[j,k] at V[j*4+k] for j=1,2 and k=0,1,2,3
    # C[0,k] at W[k*3+0] = W[3k]
    for j in range(1, 3):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Block 3: A[1:3,0] * B[0,:] -> C[1:3,0]
    # A[i,0] at U[i*3+0] = U[3i] for i=1,2
    # B[0,k] at V[0*4+k] = V[k] for k=0,1,2,3
    # C[i,0] at W[0*3+i] = W[i] for i=1,2
    for i in range(1, 3):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[0 * n + i][col] = 1
            col += 1
    
    # Block 4: A[1:3,1:3] * B[1:3,1:3] -> C[1:3,1:3]
    # This is a 2x2 * 2x2 multiplication using Strassen (rank 7)
    # A[1,1] at U[1*3+1] = U[4], A[1,2] at U[1*3+2] = U[5]
    # A[2,1] at U[2*3+1] = U[7], A[2,2] at U[2*3+2] = U[8]
    # B[1,1] at V[1*4+1] = V[5], B[1,2] at V[1*4+2] = V[6]
    # B[2,1] at V[2*4+1] = V[9], B[2,2] at V[2*4+2] = V[10]
    # C[1,1] at W[1*3+1] = W[4], C[1,2] at W[2*3+1] = W[7]
    # C[2,1] at W[1*3+2] = W[5], C[2,2] at W[2*3+2] = W[8]
    
    p00_u = 4   # A[1,1]
    p01_u = 5   # A[1,2]
    p10_u = 7   # A[2,1]
    p11_u = 8   # A[2,2]
    
    q00_v = 5   # B[1,1]
    q01_v = 6   # B[1,2]
    q10_v = 9   # B[2,1]
    q11_v = 10  # B[2,2]
    
    r00_w = 4   # C[1,1]
    r01_w = 7   # C[1,2]
    r10_w = 5   # C[2,1]
    r11_w = 8   # C[2,2]
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (P00 + P11)(Q00 + Q11)
    U[p00_u][col] += 1; U[p11_u][col] += 1
    V[q00_v][col] += 1; V[q11_v][col] += 1
    W[r00_w][col] += 1; W[r11_w][col] += 1
    col += 1
    
    # M2 = (P10 + P11)Q00
    U[p10_u][col] += 1; U[p11_u][col] += 1
    V[q00_v][col] += 1
    W[r10_w][col] += 1
    col += 1
    
    # M3 = P00(Q01 - Q11)
    U[p00_u][col] += 1
    V[q01_v][col] += 1; V[q11_v][col] += -1
    W[r00_w][col] += 1; W[r11_w][col] += 1
    col += 1
    
    # M4 = P11(Q10 - Q00)
    U[p11_u][col] += 1
    V[q10_v][col] += 1; V[q00_v][col] += -1
    W[r00_w][col] += -1; W[r01_w][col] += 1
    col += 1
    
    # M5 = P10(Q11 + Q00)
    U[p10_u][col] += 1
    V[q11_v][col] += 1; V[q00_v][col] += 1
    W[r01_w][col] += 1
    col += 1
    
    # M6 = (P11 - P10)(Q10 + Q11)
    U[p11_u][col] += 1; U[p10_u][col] += -1
    V[q10_v][col] += 1; V[q11_v][col] += 1
    W[r11_w][col] += 1
    col += 1
    
    # M7 = (P00 - P10)(Q01 + Q10)
    U[p00_u][col] += 1; U[p10_u][col] += -1
    V[q01_v][col] += 1; V[q10_v][col] += 1
    W[r00_w][col] += -1; W[r10_w][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
