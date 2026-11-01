"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    # Standard Strassen 7 products for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+c)g, m6=(c+d)(f-h), m7=(a+d)(g-f)
    # C00=m1+m4-m5+m7, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m6
    
    # U patterns for (a,b,c,d) = (A00,A01,A10,A11)
    U_pats = [
        [1,0,0,1],  # m1: a+d
        [0,0,1,1],  # m2: c+d
        [1,0,0,0],  # m3: a
        [0,0,0,1],  # m4: d
        [1,1,0,0],  # m5: a+c  -- WAIT, should be a+c where c=A10
        [0,0,1,1],  # m6: c+d
        [1,0,0,1],  # m7: a+d
    ]
    
    # Hmm, m5 = (a+c)g where c is A10 (row 1, col 0). So U for m5 = (1,0,1,0)
    U_pats[4] = [1,0,1,0]
    
    # V patterns for (e,f,g,h) = (B00,B01,B10,B11)
    V_pats = [
        [1,0,0,1],  # m1: e+h
        [1,0,0,0],  # m2: e
        [0,1,0,-1], # m3: f-h
        [0,0,1,-1], # m4: g-e
        [0,0,1,0],  # m5: g
        [0,1,0,-1], # m6: f-h
        [0,0,1,-1], # m7: g-f
    ]
    
    # W patterns for (C00,C10,C01,C11) in trace order W[k*n+i]
    # C00->W[0], C10->W[1], C01->W[2], C11->W[3]
    W_pats = [
        [1,0,0,1],  # m1: C00+1, C11+1
        [0,1,0,-1], # m2: C10+1, C11-1
        [0,0,1,1],  # m3: C01+1, C11+1
        [1,1,0,0],  # m4: C00+1, C10+1
        [-1,0,1,0], # m5: C00-1, C01+1
        [0,0,0,1],  # m6: C11+1
        [1,0,0,-1], # m7: C00+1, C11-1
    ]
    
    for block in range(2):
        base = block * 7
        vbase = block * 4
        wbase = block * 4
        for s in range(7):
            col = base + s
            for i in range(4):
                U[i][col] = U_pats[s][i]
            for j in range(4):
                V[vbase+j][col] = V_pats[s][j]
            for k in range(4):
                W[wbase+k][col] = W_pats[s][k]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = _start + 140  # 10s safety margin
    
    # Try rank-14 Strassen
    result = _strassen_rank14()
    if _verify(result["U"], result["V"], result["W"], 2, 2, 4):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
