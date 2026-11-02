import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    col = 0
    
    def strassen_2x2x2(u_base, v_base, w_base, offset):
        # u_base: list of 4 U indices for A[0,0], A[0,1], A[1,0], A[1,1]
        # v_base: list of 4 V indices for B[0,0], B[0,1], B[1,0], B[1,1]
        # w_base: list of 4 W indices for C[0,0], C[0,1], C[1,0], C[1,1]
        # offset: starting column index
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[u_base[0]][col] += 1
        U[u_base[3]][col] += 1
        V[v_base[0]][col] += 1
        V[v_base[3]][col] += 1
        W[w_base[0]][col] += 1 # C00
        W[w_base[3]][col] += 1 # C11
        col += 1
        
        # M2 = (A10+A11)B00
        U[u_base[2]][col] += 1
        U[u_base[3]][col] += 1
        V[v_base[0]][col] += 1
        W[w_base[2]][col] += 1 # C10
        W[w_base[3]][col] += -1 # C11
        col += 1
        
        # M3 = A00(B01-B11)
        U[u_base[0]][col] += 1
        V[v_base[1]][col] += 1
        V[v_base[3]][col] += -1
        W[w_base[1]][col] += 1 # C01
        W[w_base[3]][col] += 1 # C11
        col += 1
        
        # M4 = A11(B10-B00)
        U[u_base[3]][col] += 1
        V[v_base[2]][col] += 1
        V[v_base[0]][col] += -1
        W[w_base[0]][col] += -1 # C00
        W[w_base[2]][col] += 1 # C10
        col += 1
        
        # M5 = (A00+A01)B11
        U[u_base[0]][col] += 1
        U[u_base[1]][col] += 1
        V[v_base[3]][col] += 1
        W[w_base[1]][col] += 1 # C01
        W[w_base[3]][col] += -1 # C11
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_base[2]][col] += 1
        U[u_base[0]][col] += -1
        V[v_base[0]][col] += 1
        V[v_base[1]][col] += 1
        W[w_base[3]][col] += 1 # C11
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_base[1]][col] += 1
        U[u_base[3]][col] += -1
        V[v_base[2]][col] += 1
        V[v_base[3]][col] += 1
        W[w_base[0]][col] += 1 # C00
        col += 1

    # Block 1: A_left (2x2) * B_top_left (2x2) -> C_left (2x2)
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    u1 = [0, 1, 3, 4]
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    v1 = [0, 1, 5, 6]
    # W indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    w1 = [0, 1, 2, 3]
    strassen_2x2x2(u1, v1, w1, col)
    
    # Block 2a: A_left (2x2) * B_col2 (2x1) -> C_col2 (2x1)
    # U indices: same u1
    # V indices: B[0,2]=2, B[1,2]=7
    # W indices: C[0,2]=4, C[1,2]=5
    # Terms:
    # A00 B02 -> C02
    U[0][col] += 1
    V[2][col] += 1
    W[4][col] += 1
    col += 1
    
    # A01 B12 -> C02
    U[1][col] += 1
    V[7][col] += 1
    W[4][col] += 1
    col += 1
    
    # A10 B02 -> C12
    U[3][col] += 1
    V[2][col] += 1
    W[5][col] += 1
    col += 1
    
    # A11 B12 -> C12
    U[4][col] += 1
    V[7][col] += 1
    W[5][col] += 1
    col += 1
    
    # Block 2b: A_left (2x2) * B_cols34 (2x2) -> C_cols34 (2x2)
    # U indices: same u1
    # V indices: B[0,3]=3, B[0,4]=4, B[1,3]=8, B[1,4]=9
    v2b = [3, 4, 8, 9]
    # W indices: C[0,3]=6, C[0,4]=7, C[1,3]=8, C[1,4]=9
    w2b = [6, 7, 8, 9]
    strassen_2x2x2(u1, v2b, w2b, col)
    
    # Block 3: A_right (2x1) * B_bottom (1x5) -> C_all (2x5)
    # U indices: A[0,2]=2, A[1,2]=5
    # V indices: B[2,0]=10, ..., B[2,4]=14
    # W indices: C[0,k]=2k, C[1,k]=2k+1
    
    for k in range(5):
        # A02 B2k -> C0k
        U[2][col] += 1
        V[10+k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        # A12 B2k -> C1k
        U[5][col] += 1
        V[10+k][col] += 1
        W[2*k+1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
