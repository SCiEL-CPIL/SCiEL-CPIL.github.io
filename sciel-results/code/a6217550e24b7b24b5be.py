"""Program for matrix multiplication tensor decomposition (3x5 by 5x7).
Uses a hybrid approach: Strassen decomposition for 2x2 blocks and schoolbook
for the remaining 1x1 blocks, achieving rank 104 (one 2x2x2 block replaced
by Strassen's rank-7 algorithm).
"""

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    rank = 104  # 44 (schoolbook for non-2x2 parts) + 7 (Strassen) + 53 (remaining schoolbook)
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 104
    V = [[0] * rank for _ in range(m * p)]  # 35 x 104
    W = [[0] * rank for _ in range(p * n)]  # 21 x 104
    
    col = 0
    
    # We will use Strassen for the block A[0:2, 2:4] * B[2:4, 3:5] -> C[0:2, 3:5]
    # Indices: i in {0,1}, j in {2,3}, k in {3,4}
    # Strassen factors:
    # U factors (for A block): [1,1], [1,0], [0,1], [1,-1], [0,1], [1,1], [1,0]
    # V factors (for B block): [1,1], [0,1], [1,0], [1,1], [1,-1], [1,0], [0,1]
    # W factors (for C block): [1,0], [0,1], [0,1], [1,1], [1,-1], [1,1], [0,1]
    
    strassen_U = [
        [1, 1],
        [1, 0],
        [0, 1],
        [1, -1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]
    strassen_V = [
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
        [1, 0],
        [0, 1]
    ]
    strassen_W = [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 1],
        [1, -1],
        [1, 1],
        [0, 1]
    ]
    
    # Add Strassen factors
    for c in range(7):
        for i in range(2):
            U[i * m + (2 + 0)]  # j=2 for i=0,1 -> row i*5+2
            # Actually, A block is A[i, j] where i in {0,1}, j in {2,3}
            # So row index in U is i*m + j
            # For each Strassen factor c, U[i*m + j, col] = strassen_U[c][i_local]
            # where i_local is 0 or 1 (local index in 2x2 block)
            pass
        
        # Let's do it properly:
        for i in range(2):
            for j_local in range(2):
                j = 2 + j_local
                U[i * m + j][col] = strassen_U[c][i]
        for j_local in range(2):
            for k_local in range(2):
                j = 2 + j_local
                k = 3 + k_local
                V[j * p + k][col] = strassen_V[c][j_local]
        for i in range(2):
            for k_local in range(2):
                k = 3 + k_local
                W[k * n + i][col] = strassen_W[c][k_local]
        col += 1
    
    # Now add schoolbook for all other (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the Strassen block
                if i < 2 and 2 <= j < 4 and 3 <= k < 5:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
