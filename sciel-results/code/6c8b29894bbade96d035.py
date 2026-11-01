import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = 2, 2, 11
    
    # Rank 39: 5 Strassen blocks (rank 7) + 1 Schoolbook block (rank 4)
    rank = 5 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Indices for A (2x2): 0=a, 1=b, 2=c, 3=d
    # Indices for B (2x11): 0..10 for row 0, 11..21 for row 1
    # V index for B[j, k] = j*p + k
    
    # W indices (C indices)
    # C is 2x11. C[i, k].
    # Trace convention: W[k*n + i, r].
    # n=2. W[2*k + i, r].
    
    for block in range(5):
        if time.time() > deadline - 5:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        
        # B indices
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # Strassen terms
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

    # Last column k2 = 10
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # Schoolbook for 2x2 * 2x1
    # C[0, k2] = a B[0, k2] + b B[1, k2]
    # C[1, k2] = c B[0, k2] + d B[1, k2]
    
    # Term 1: a * B[0, k2]
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    # Term 2: b * B[1, k2]
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    # Term 3: c * B[0, k2]
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    # Term 4: d * B[1, k2]
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
