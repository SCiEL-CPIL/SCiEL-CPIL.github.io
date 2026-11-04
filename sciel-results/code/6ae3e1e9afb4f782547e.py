import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Validate tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition via 2x2 Strassen + schoolbook 2x2x2. Rank 56."""
    n, m, p = 4, 4, 4
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 2x2x2 coefficients
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11), M4=A11(B10-B00)
    # M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    strassen_U = [
        [1,0,1,0,1,-1,0],  # A00
        [0,0,0,0,1,0,1],   # A01
        [0,1,0,0,0,1,0],   # A10
        [1,1,0,1,0,0,-1],  # A11
    ]
    strassen_V = [
        [1,1,0,0,0,1,0],   # B00
        [0,0,1,0,0,1,0],   # B01
        [0,0,0,1,0,0,1],   # B10
        [1,0,-1,-1,1,0,1], # B11
    ]
    strassen_W = [
        [1,1,0,1,-1,0,1],  # C00 (k=0,i=0)
        [0,0,1,0,1,0,0],   # C01 (k=0,i=1)
        [0,1,0,1,0,0,0],   # C10 (k=1,i=0)
        [1,-1,1,0,0,1,0],  # C11 (k=1,i=1)
    ]
    
    # For each Strassen term r2 (0..6) and each schoolbook 2x2x2 term s (0..7):
    # s encodes (ri, cj, rk) where ri,cj,rk in {0,1}
    for r2 in range(7):
        for s in range(8):
            col = r2 * 8 + s
            ri = s // 4
            cj = (s % 4) // 2
            rk = s % 2
            
            # U: A[i,j] where i=2*i2+ri, j=2*j2+cj
            for i2 in range(2):
                for j2 in range(2):
                    c = strassen_U[i2*2+j2][r2]
                    if c != 0:
                        i = 2*i2 + ri
                        j = 2*j2 + cj
                        U[i*4+j][col] += c
                        
            # V: B[j,k] where j=2*j2+cj, k=2*k2+rk
            for j2 in range(2):
                for k2 in range(2):
                    c = strassen_V[j2*2+k2][r2]
                    if c != 0:
                        j = 2*j2 + cj
                        k = 2*k2 + rk
                        V[j*4+k][col] += c
                        
            # W: C[k,i] where k=2*k2+rk, i=2*i2+ri
            for k2 in range(2):
                for i2 in range(2):
                    c = strassen_W[k2*2+i2][r2]
                    if c != 0:
                        k = 2*k2 + rk
                        i = 2*i2 + ri
                        W[k*4+i][col] += c
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    
    try:
        U4, V4, W4 = _build_4x4x4_strassen()
        if _validate(U4, V4, W4, 4, 4, 4):
            rank_s = len(U4[0])
            rank_sb = 4 * 4 * 5
            U = [row[:] for row in U4]
            V = [row[:] for row in V4]
            W = [row[:] for row in W4]
            for i in range(4):
                U[i].extend([0]*rank_sb)
            for j in range(4):
                V[j].extend([0]*rank_sb)
            for k in range(9):
                W[k].extend([0]*rank_sb)
            col = rank_s
            for i in range(4):
                for j in range(4):
                    for k in range(4, 9):
                        U[i*4+j][col] = 1
                        V[j*9+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
            if _validate(U, V, W, 4, 4, 9):
                if time.time() - start_time < 140:
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
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
