import numpy as np

DIMENSIONS = [2, 5, 5]

def _winograd_f2():
    """
    Returns U, V, W for 2x2 by 2x2 matrix multiplication using Winograd F(2).
    Rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a+b)(e+g) -> a=e=g=1, b=1
    # a=0, b=1, c=0, d=0
    # e=0, f=0, g=0, h=0
    # U[0]=1, U[1]=1
    # V[2]=1, V[3]=1 (e=0, f=0, g=0, h=0? No, e is V[0*2+0]=V[0], g is V[1*2+0]=V[2])
    # Let's map indices carefully.
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[c00, c01], [c10, c11]] -> indices 0,1,2,3
    
    # M1 = (a+b)(e+g)
    # a: U[0]=1, b: U[1]=1
    # e: V[0]=1, g: V[2]=1
    # c00: W[0]=1, c01: W[1]=-1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[1][0] = -1
    
    # M2 = (a-b)(e-g)
    # a: U[0]=1, b: U[1]=-1
    # e: V[0]=1, g: V[2]=-1
    # c00: W[0]=1, c01: W[1]=1
    U[0][1] = 1; U[1][1] = -1
    V[0][1] = 1; V[2][1] = -1
    W[0][1] = 1; W[1][1] = 1
    
    # M3 = (c+d)(g+h)
    # c: U[2]=1, d: U[3]=1
    # g: V[2]=1, h: V[3]=1
    # c10: W[2]=1, c11: W[3]=-1
    U[2][2] = 1; U[3][2] = 1
    V[2][2] = 1; V[3][2] = 1
    W[2][2] = 1; W[3][2] = -1
    
    # M4 = (c-d)(g-h)
    # c: U[2]=1, d: U[3]=-1
    # g: V[2]=1, h: V[3]=-1
    # c10: W[2]=1, c11: W[3]=1
    U[2][3] = 1; U[3][3] = -1
    V[2][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = a h
    # a: U[0]=1
    # h: V[3]=1
    # c10: W[2]=1
    U[0][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    
    # M6 = b f
    # b: U[1]=1
    # f: V[1]=1
    # c01: W[1]=1
    U[1][5] = 1
    V[1][5] = 1
    W[1][5] = 1
    
    # M7 = d e
    # d: U[3]=1
    # e: V[0]=1
    # c00: W[0]=1, c11: W[3]=1
    U[3][6] = 1
    V[0][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will build the full U, V, W by concatenating sub-decompositions.
    # We need to map local indices to global indices.
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_decomp(U_loc, V_loc, W_loc, i_off, j_off, k_off):
        """
        Adds a local decomposition to the global ones.
        i_off: offset for i index in U and W
        j_off: offset for j index in U and V
        k_off: offset for k index in V and W
        """
        nonlocal total_rank
        rank_loc = len(U_loc[0])
        U_new = [[0]*total_rank for _ in range(n*m)]
        V_new = [[0]*total_rank for _ in range(m*p)]
        W_new = [[0]*total_rank for _ in range(p*n)]
        
        # Copy existing parts
        # This is inefficient, but rank is small.
        # Better to accumulate in lists of rows.
        
    # Let's use lists of rows to accumulate.
    U_rows = [[] for _ in range(n*m)]
    V_rows = [[] for _ in range(m*p)]
    W_rows = [[] for _ in range(p*n)]
    
    def add_block(U_loc, V_loc, W_loc, i_off, j_off, k_off):
        rank_loc = len(U_loc[0])
        for r in range(rank_loc):
            for i_loc in range(len(U_loc)):
                val = U_loc[i_loc][r]
                if val != 0:
                    i_g = i_off + i_loc
                    U_rows[i_g].append(val)
            for j_loc in range(len(V_loc)):
                val = V_loc[j_loc][r]
                if val != 0:
                    j_g = j_off + j_loc
                    V_rows[j_g].append(val)
            for k_loc in range(len(W_loc)):
                val = W_loc[k_loc][r]
                if val != 0:
                    k_g = k_off + k_loc
                    W_rows[k_g].append(val)
        # Note: We are appending column by column.
        # The length of each row list should be the same at the end.
        
    # Block 1: A1 (2x4), B1 (4x4), C1 (2x4)
    # i: 0..1, j: 0..3, k: 0..3
    # Split B1 into Ba (4x2) and Bb (4x2)
    # A1 Ba: i 0..1, j 0..3, k 0..1
    # A1 Bb: i 0..1, j 0..3, k 2..3
    
    # A1 Ba: Split A1 into A11 (2x2) and A12 (2x2)
    # A11 Ba_top (2x2): i 0..1, j 0..1, k 0..1
    U_w, V_w, W_w = _winograd_f2()
    add_block(U_w, V_w, W_w, i_off=0, j_off=0, k_off=0)
    
    # A12 Ba_bot (2x2): i 0..1, j 2..3, k 0..1
    add_block(U_w, V_w, W_w, i_off=0, j_off=2, k_off=0)
    
    # A1 Bb: Split A1 into A11 and A12
    # A11 Bb_top (2x2): i 0..1, j 0..1, k 2..3
    add_block(U_w, V_w, W_w, i_off=0, j_off=0, k_off=2)
    
    # A12 Bb_bot (2x2): i 0..1, j 2..3, k 2..3
    add_block(U_w, V_w, W_w, i_off=0, j_off=2, k_off=2)
    
    # Block 2: A2 (2x1), B3 (1x4), C1a (2x4)
    # A2 is i 0..1, j 4. B3 is j 4, k 0..3.
    # This is 2x1 by 1x4.
    U_s, V_s, W_s = _schoolbook(2, 1, 4)
    add_block(U_s, V_s, W_s, i_off=0, j_off=4, k_off=0)
    
    # Block 3: A1 (2x4), B2 (4x1), C2a (2x1)
    # A1 is i 0..1, j 0..3. B2 is j 0..3, k 4.
    # This is 2x4 by 4x1.
    U_s, V_s, W_s = _schoolbook(2, 4, 1)
    add_block(U_s, V_s, W_s, i_off=0, j_off=0, k_off=4)
    
    # Block 4: A2 (2x1), B4 (1x1), C2b (2x1)
    # A2 is i 0..1, j 4. B4 is j 4, k 4.
    # This is 2x1 by 1x1.
    U_s, V_s, W_s = _schoolbook(2, 1, 1)
    add_block(U_s, V_s, W_s, i_off=0, j_off=4, k_off=4)
    
    # Determine total rank
    if U_rows[0]:
        total_rank = len(U_rows[0])
    else:
        total_rank = 0
        
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r in range(total_rank):
            U[i][r] = U_rows[i][r]
    for j in range(m*p):
        for r in range(total_rank):
            V[j][r] = V_rows[j][r]
    for k in range(p*n):
        for r in range(total_rank):
            W[k][r] = W_rows[k][r]
            
    return {"U": U, "V": V, "W": W}
