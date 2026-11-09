import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for j in [j0, j1] and k in [k0, k1]
    # j0, j1 are indices in the middle dimension (0, 1 for the 2x2 part)
    # k0, k1 are indices in the output dimension
    def add_strassen(j0, j1, k0, k1):
        nonlocal col
        # 7 terms
        # 1: (A00 + A11)(B00 + B11) -> C00 + C11
        # U: A00, A11
        U[j0][col] = 1 # i=0, j=j0
        U[j1 + n][col] = 1 # i=1, j=j1
        # V: B00, B11
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        # W: C00 (i=0, k=k0), C11 (i=1, k=k1)
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # 2: (A10 + A11)B00 -> C10
        # U: A10, A11
        U[j0 + n][col] = 1
        U[j1 + n][col] = 1
        # V: B00
        V[j0 * p + k0][col] = 1
        # W: C10 (i=1, k=k0)
        W[k0 * n + 1][col] = 1
        col += 1
        
        # 3: A00(B01 - B11) -> C01
        # U: A00
        U[j0][col] = 1
        # V: B01, -B11
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        # W: C01 (i=0, k=k1)
        W[k1 * n + 0][col] = 1
        col += 1
        
        # 4: A11(B10 - B00) -> C00, C10
        # U: A11
        U[j1 + n][col] = 1
        # V: B10, -B00
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        # W: C00 (i=0, k=k0), C10 (i=1, k=k0)
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # 5: (A00 + A01)B11 -> C01, C11
        # U: A00, A01
        U[j0][col] = 1
        U[j1][col] = 1
        # V: B11
        V[j1 * p + k1][col] = 1
        # W: C01 (i=0, k=k1), C11 (i=1, k=k1)
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # 6: (A10 - A00)(B00 + B01) -> C11
        # U: A10, -A00
        U[j0 + n][col] = 1
        U[j0][col] = -1
        # V: B00, B01
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        # W: C11 (i=1, k=k1)
        W[k1 * n + 1][col] = 1
        col += 1
        
        # 7: (A01 - A11)(B10 + B11) -> C00, C10
        # U: A01, -A11
        U[j1][col] = 1
        U[j1 + n][col] = -1
        # V: B10, B11
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        # W: C00 (i=0, k=k0), C10 (i=1, k=k0)
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

    # 4 Strassen blocks for j in {0,1}
    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    add_strassen(0, 1, 6, 7)
    
    # 1 Schoolbook block for j in {0,1}, k=8
    # C_i,8 = A_i,0 B_0,8 + A_i,1 B_1,8
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
            
    # 1 Rank-1 block for j=2, k in 0..8
    # T_i,2,k = 1
    # U_i,2 = 1
    # V_2,k = 1
    # W_k,i = 1
    for i in range(2):
        U[i * m + 2][col] = 1
    for k in range(9):
        V[2 * p + k][col] = 1
    for k in range(9):
        for i in range(2):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
