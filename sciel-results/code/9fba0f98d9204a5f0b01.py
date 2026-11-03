"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+d)(e+h)
    # M6 = (b-d)(g+h)
    # M7 = (c-a)(e+f)
    #
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M3
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 - M4
    
    # For 2x2x2, n=2, m=2, p=2
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Indexing: U[i*m+j, c] where i,j in {0,1}
    # Row 0: (0,0)=a, Row 1: (0,1)=b, Row 2: (1,0)=c, Row 3: (1,1)=d
    # V: Row 0: (0,0)=e, Row 1: (0,1)=f, Row 2: (1,0)=g, Row 3: (1,1)=h
    # W: Row 0: (0,0)=C11, Row 1: (0,1)=C12, Row 2: (1,0)=C21, Row 3: (1,1)=C22
    
    # M1 = (a+b)(e+f)
    U[0, 0] = 1  # a
    U[1, 0] = 1  # b
    V[0, 0] = 1  # e
    V[1, 0] = 1  # f
    W[0, 0] = 1  # C11
    W[1, 0] = 1  # C12
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22
    
    # M4 = d(g-e)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e
    W[0, 3] = -1 # C11
    W[3, 3] = -1 # C22
    
    # M5 = (a+d)(e+h)
    U[0, 4] = 1  # a
    U[3, 4] = 1  # d
    V[0, 4] = 1  # e
    V[3, 4] = 1  # h
    W[0, 4] = 1  # C11
    W[2, 4] = 1  # C21
    
    # M6 = (b-d)(g+h)
    U[1, 5] = 1  # b
    U[3, 5] = -1 # d
    V[2, 5] = 1  # g
    V[3, 5] = 1  # h
    W[3, 5] = 1  # C22
    
    # M7 = (c-a)(e+f)
    U[2, 6] = 1  # c
    U[0, 6] = -1 # a
    V[0, 6] = 1  # e
    V[1, 6] = 1  # f
    W[0, 6] = -1 # C11
    W[2, 6] = 1  # C21
    
    return U.tolist(), V.tolist(), W.tolist()

def _embed_strassen_in_3x4x7():
    """Embed Strassen 2x2x2 decompositions into 3x4x7."""
    n, m, p = 3, 4, 7
    
    # We'll use Strassen for the 2x2x2 sub-blocks
    # Split: rows 0-1, cols 0-1, result cols 0-1
    # This is a 2x2x2 sub-problem
    
    U_strassen, V_strassen, W_strassen = _strassen_2x2()
    
    # For the full 3x4x7, we need to embed this
    # The 2x2x2 sub-problem uses:
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    
    # Full U has shape (3*4, rank) = (12, rank)
    # Full V has shape (4*7, rank) = (28, rank)
    # Full W has shape (7*3, rank) = (21, rank)
    
    # For now, let's just use the trivial decomposition and see if we can
    # replace some 2x2x2 blocks with Strassen
    
    rank = n * m * p  # 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Use trivial rank-84 decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
