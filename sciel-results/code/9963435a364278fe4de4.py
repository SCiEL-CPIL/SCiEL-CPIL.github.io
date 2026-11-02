import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen blocks
    # 5 blocks of 2x2x2
    for l in range(5):
        base_col = l * 7
        # s=0: m1 = (a+d)(e+h)
        c = base_col + 0
        U[0][c] = 1; U[4][c] = 1
        V[2*l][c] = 1; V[10+2*l+1][c] = 1
        W[4*l][c] = 1; W[4*l+3][c] = 1
        
        # s=1: m2 = (c+d)e
        c = base_col + 1
        U[3][c] = 1; U[4][c] = 1
        V[2*l][c] = 1
        W[4*l+1][c] = 1
        
        # s=2: m3 = a(f-h)
        c = base_col + 2
        U[0][c] = 1
        V[2*l+1][c] = 1; V[10+2*l+1][c] = -1
        W[4*l+2][c] = 1
        
        # s=3: m4 = d(g-e)
        c = base_col + 3
        U[4][c] = 1
        V[10+2*l][c] = 1; V[2*l][c] = -1
        W[4*l][c] = 1; W[4*l+1][c] = 1
        
        # s=4: m5 = (a+b)h
        c = base_col + 4
        U[0][c] = 1; U[1][c] = 1
        V[10+2*l+1][c] = 1
        W[4*l][c] = -1; W[4*l+2][c] = 1
        
        # s=5: m6 = (c-a)(e+f)
        c = base_col + 5
        U[3][c] = 1; U[0][c] = -1
        V[2*l][c] = 1; V[2*l+1][c] = 1
        W[4*l+3][c] = 1
        
        # s=6: m7 = (b-d)(g+h)
        c = base_col + 6
        U[1][c] = 1; U[4][c] = -1
        V[10+2*l][c] = 1; V[10+2*l+1][c] = 1
        W[4*l][c] = 1

    # Schoolbook for A12 B21
    # A12 is 2x1 (i=0,1; j=2)
    # B21 is 1x10 (j=2; k=0..9)
    col_offset = 35
    for i in range(2):
        for k in range(10):
            c = col_offset + i * 10 + k
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][c] = 1
            V[v_idx][c] = 1
            W[w_idx][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
