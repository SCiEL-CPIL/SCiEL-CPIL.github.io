"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Return (U, V, W) for 2x2 matrix multiplication with rank 7 (Strassen).
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen 7 products
    # M1 = (a00+a11)*(b00+b11)
    U[0][0]=1; U[1][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[3][0]=1
    
    # M2 = (a10+a11)*b00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # M3 = a00*(b01-b11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    
    # M4 = a11*(b10-b00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    
    # M5 = (a00+a01)*b11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W

def _build_3x3_schoolbook():
    """
    Return (U, V, W) for 3x3 matrix multiplication with rank 27 (schoolbook).
    U: 9x27, V: 9x27, W: 9x27
    """
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i*3+k][col] = 1
                V[k*3+j][col] = 1
                W[j*3+i][col] = 1
                col += 1
    
    return U, V, W

def _kron_combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two tensor decompositions via Kronecker product.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    
    N = n1 * n2
    M = m1 * m2
    P = p1 * p2
    
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            
            for i1 in range(n1):
                for k1 in range(m1):
                    u1_val = U1[i1*m1+k1][r1_idx]
                    if u1_val == 0: continue
                    for i2 in range(n2):
                        for k2 in range(m2):
                            u2_val = U2[i2*m2+k2][r2_idx]
                            if u2_val == 0: continue
                            row_u = (i1*n2 + i2) * M + (k1*m2 + k2)
                            U[row_u][col] += u1_val * u2_val
            
            for k1 in range(m1):
                for j1 in range(p1):
                    v1_val = V1[k1*p1+j1][r1_idx]
                    if v1_val == 0: continue
                    for k2 in range(m2):
                        for j2 in range(p2):
                            v2_val = V2[k2*p2+j2][r2_idx]
                            if v2_val == 0: continue
                            row_v = (k1*m2 + k2) * P + (j1*p2 + j2)
                            V[row_v][col] += v1_val * v2_val
            
            for j1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[j1*n1+i1][r1_idx]
                    if w1_val == 0: continue
                    for j2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[j2*n2+i2][r2_idx]
                            if w2_val == 0: continue
                            row_w = (j1*p2 + j2) * N + (i1*n2 + i2)
                            W[row_w][col] += w1_val * w2_val
    
    return U, V, W

def _verify_bilinear(U, V, W, n, m, p, num_tests=100, seed=42):
    """Verify decomposition."""
    random.seed(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        val = 0.0
        for r in range(len(U[0])):
            val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        if abs(val - 1.0) > 1e-10:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    
    # Strategy: Strassen (2x2) outer x Schoolbook (3x3) inner = Rank 189
    U_outer, V_outer, W_outer = _build_strassen_2x2()
    U_inner, V_inner, W_inner = _build_3x3_schoolbook()
    
    U, V, W = _kron_combine(U_outer, V_outer, W_outer, U_inner, V_inner, W_inner, 2, 2, 2, 3, 3, 3)
    
    if _verify_bilinear(U, V, W, n, m, p, num_tests=200):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook 6x6x6 (Rank 216)
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
