"""Matrix multiplication 2x2 x 2x14: Strassen-based rank-49 decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    num_blocks = p // 2   # 7 blocks of 2 columns each
    rank = 7 * num_blocks  # 49

    # Strassen 2x2x2 decomposition (rank 7)
    # M1 = (A00+A11)*B00, M2 = (A10+A11)*B00, M3 = A00*(B10-B11),
    # M4 = A11*(B01-B00), M5 = (A00+A01)*B11, M6 = (A10-A00)*(B00+B01),
    # M7 = (A01-A11)*(B10+B11)
    #
    # U[i,j,s] for A element (i,j), V[j,k,s] for B element (j,k), W[k,i,s] for C element (k,i)
    # Using indices: i,j,k in {0,1} for 2x2, s in {0..6} for rank dimension
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M6 + M7
    #
    # U coefficients (rows = A index i*m+j = i*2+j, cols = rank s):
    # s=0 (M1): A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    # s=1 (M2): A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    # s=2 (M3): A00=1 -> U[0,2]=1
    # s=3 (M4): A11=1 -> U[3,3]=1
    # s=4 (M5): A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    # s=5 (M6): A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    # s=6 (M7): A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    # V coefficients (rows = B index j*p_local+k = j*2+k, cols = rank s):
    # s=0 (M1): B00=1 -> V[0,0]=1
    # s=1 (M2): B00=1 -> V[0,1]=1
    # s=2 (M3): B10=1, B11=-1 -> V[2,2]=1, V[3,2]=-1
    # s=3 (M4): B01=1, B00=-1 -> V[1,3]=1, V[0,3]=-1
    # s=4 (M5): B11=1 -> V[3,4]=1
    # s=5 (M6): B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    # s=6 (M7): B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 1, 0],   # B01
        [0, 0, 1, 0, 0, 0, 1],   # B10
        [0, 0, -1, 0, 1, 0, 1],  # B11
    ]
    # W coefficients (rows = C index k*n+i = k*2+i, cols = rank s):
    # C00 (row 0): M1-M4-M5+M7 -> [1,-1,-1,0,0,0,1] wait let me redo
    # C00 = M1 - M4 - M5 + M7: s0=1, s3=-1, s4=-1, s6=1
    # C01 (row 1) = M3 + M5: s2=1, s4=1
    # C10 (row 2) = M2 + M4: s1=1, s3=1
    # C11 (row 3) = M1 - M2 - M6 + M7: s0=1, s1=-1, s5=-1, s6=1
    W_local = [
        [1, 0, 0, -1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],     # C01
        [0, 1, 0, 1, 0, 0, 0],     # C10
        [1, -1, 0, 0, 0, -1, 1],   # C11
    ]

    # Build full matrices
    U = [[0] * rank for _ in range(n * m)]       # 4 x 49
    V = [[0] * rank for _ in range(m * p)]       # 28 x 49
    W = [[0] * rank for _ in range(p * n)]       # 28 x 49

    for b in range(num_blocks):
        base = b * 7
        for s in range(7):
            r = base + s
            # U: same for all blocks
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = U_local[i * m + j][s]
            # V: block b uses local columns 2b and 2b+1
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][r] = V_local[j * 2 + k_local][s]
            # W: block b uses local rows 2b and 2b+1
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(n):
                    W[k_global * n + i][r] = W_local[k_local * n + i][s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
