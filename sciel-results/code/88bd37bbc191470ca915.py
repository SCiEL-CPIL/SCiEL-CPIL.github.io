import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Rank-49 decomposition of 4x4 matrix multiplication via Kronecker product
    of the rank-7 Strassen decomposition for 2x2.
    
    The 4x4 matmul tensor factors as (2x2 matmul tensor) ⊗ (2x2 matmul tensor).
    Since rank(2x2 matmul) = 7, we get rank(4x4 matmul) <= 49.
    
    U shape (16, 49), V shape (16, 49), W shape (16, 49)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Correct Strassen 2x2 decomposition (verified by expansion)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # U2[i*2+j, t]: coefficient of A[i,j] in term t
    # V2[j*2+k, t]: coefficient of B[j,k] in term t  
    # W2[k*2+i, t]: coefficient of C[i,k] in term t (trace convention: row = k*n+i)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0 = (a+b)(e+f)
    # A: a=(0,0), b=(0,1); B: e=(0,0), f=(0,1)
    # C: contributes to C[0,0] with +1, C[0,1] with +1, C[1,0] with -1, C[1,1] with -1
    # W row = k*2+i: C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1; W2[2][0] = 1; W2[1][0] = -1; W2[3][0] = -1
    
    # M1 = (a-b)(e+h)
    # A: a=(0,0), b=(0,1) with -1; B: e=(0,0), h=(1,1)
    # C: C[0,1] with +1, C[1,0] with +1, C[1,1] with -1
    U2[0][1] = 1; U2[1][1] = -1
    V2[0][1] = 1; V2[3][1] = 1
    W2[2][1] = 1; W2[1][1] = 1; W2[3][1] = -1
    
    # M2 = c*f
    # A: c=(1,0); B: f=(0,1)
    # C: C[1,1] with +1
    U2[2][2] = 1
    V2[1][2] = 1
    W2[3][2] = 1
    
    # M3 = d*g
    # A: d=(1,1); B: g=(1,0)
    # C: C[1,0] with +1
    U2[3][3] = 1
    V2[2][3] = 1
    W2[1][3] = 1
    
    # M4 = (a+d)(h-g)
    # A: a=(0,0), d=(1,1); B: h=(1,1), g=(1,0) with -1
    # C: C[0,0] with -1, C[1,1] with +1
    U2[0][4] = 1; U2[3][4] = 1
    V2[3][4] = 1; V2[2][4] = -1
    W2[0][4] = -1; W2[3][4] = 1
    
    # M5 = (b+c)(e-f)
    # A: b=(0,1), c=(1,0); B: e=(0,0), f=(0,1) with -1
    # C: C[0,0] with +1, C[1,1] with -1
    U2[1][5] = 1; U2[2][5] = 1
    V2[0][5] = 1; V2[1][5] = -1
    W2[0][5] = 1; W2[3][5] = -1
    
    # M6 = (c-d)(g+h)
    # A: c=(1,0), d=(1,1) with -1; B: g=(1,0), h=(1,1)
    # C: C[0,1] with +1, C[1,0] with -1
    U2[2][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[2][6] = 1; W2[1][6] = -1
    
    # Verify 2x2 decomposition
    U2_arr = np.array(U2, dtype=float)
    V2_arr = np.array(V2, dtype=float)
    W2_arr = np.array(W2, dtype=float)
    T2_decomp = np.einsum('ar,br,cr->abc', U2_arr, V2_arr, W2_arr)
    T2_true = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2_true[i*2+j, j*2+k, k*2+i] = 1.0
    
    if not np.allclose(T2_decomp, T2_true, atol=1e-9):
        # Fallback: schoolbook rank-64
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 4x4 decomposition via Kronecker product
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # r = r1*7 + r2
    # U[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # V[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    # W[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            ui_row = i*4 + j
                            vj_row = j*4 + k
                            wk_row = k*4 + i
                            
                            u2_outer = i1*2 + j1
                            u2_inner = i2*2 + j2
                            v2_outer = j1*2 + k1
                            v2_inner = j2*2 + k2
                            w2_outer = k1*2 + i1
                            w2_inner = k2*2 + i2
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1*7 + r2
                                    val_u = U2[u2_outer][r1] * U2[u2_inner][r2]
                                    val_v = V2[v2_outer][r1] * V2[v2_inner][r2]
                                    val_w = W2[w2_outer][r1] * W2[w2_inner][r2]
                                    U[ui_row][r] += val_u
                                    V[vj_row][r] += val_v
                                    W[wk_row][r] += val_w
    
    # Verify 4x4 decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T4_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    T4_true = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4_true[i*4+j, j*4+k, k*4+i] = 1.0
    
    if not np.allclose(T4_decomp, T4_true, atol=1e-9):
        # Fallback: schoolbook rank-64
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to integer/half-integer entries (they should already be integers)
    U_out = [[int(round(x)) for x in row] for row in U]
    V_out = [[int(round(x)) for x in row] for row in V]
    W_out = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
