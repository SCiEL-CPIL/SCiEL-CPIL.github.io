import time

_START_TIME = time.time()
_TIMEOUT = 120

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    Uses Strassen's algorithm which achieves rank 7 for 2x2 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    
    U has shape (n*m, rank) = (4, 7)
    V has shape (m*p, rank) = (4, 7)  
    W has shape (p*n, rank) = (4, 7)
    """
    if time.time() - _START_TIME > _TIMEOUT - 1:
        raise TimeoutError("Time limit exceeded")
        
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U factors (row = i*m+j, col = r)
    # r=0: M1: a, d
    U[0][0] = 1
    U[3][0] = 1
    # r=1: M2: c, d
    U[2][1] = 1
    U[3][1] = 1
    # r=2: M3: a
    U[0][2] = 1
    # r=3: M4: d
    U[3][3] = 1
    # r=4: M5: a, b
    U[0][4] = 1
    U[1][4] = 1
    # r=5: M6: c, -a
    U[2][5] = 1
    U[0][5] = -1
    # r=6: M7: b, -d
    U[1][6] = 1
    U[3][6] = -1
    
    # V factors (row = j*p+k, col = r)
    # r=0: M1: e, h
    V[0][0] = 1
    V[3][0] = 1
    # r=1: M2: e
    V[0][1] = 1
    # r=2: M3: f, -h
    V[1][2] = 1
    V[3][2] = -1
    # r=3: M4: g, -e
    V[2][3] = 1
    V[0][3] = -1
    # r=4: M5: h
    V[3][4] = 1
    # r=5: M6: e, f
    V[0][5] = 1
    V[1][5] = 1
    # r=6: M7: g, h
    V[2][6] = 1
    V[3][6] = 1
    
    # W factors (row = k*n+i, col = r)
    # C[0,0] (k=0, i=0) -> W[0]: M1 + M4 - M5 + M7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # C[1,0] (k=0, i=1) -> W[1]: M2 + M4
    W[1][1] = 1
    W[1][3] = 1
    
    # C[0,1] (k=1, i=0) -> W[2]: M3 + M5
    W[2][2] = 1
    W[2][4] = 1
    
    # C[1,1] (k=1, i=1) -> W[3]: M1 - M2 + M3 + M6
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}
