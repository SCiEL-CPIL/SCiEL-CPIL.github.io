import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s (150s kill)
    
    n, m, p = DIMENSIONS
    
    # Rank 46 decomposition using Strassen on 2x2x2 blocks
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for J={ja, jb}, K={ka, kb}
    def add_strassen(ja, jb, ka, kb):
        nonlocal col
        # Local A indices: 0->ja, 1->jb
        # Local B indices: 0->ka, 1->kb
        # Local C indices: 0->ka, 1->kb (mapped to global k*n+i)
        
        # Strassen M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A (U), B (V), C (W)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1
        U[0*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        # C00 (k=ka, i=0) idx ka*n+0
        # C11 (k=kb, i=1) idx kb*n+1
        # C01 (k=kb, i=0) idx kb*n+0
        # C10 (k=ka, i=1) idx ka*n+1
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        W[ka*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M2
        U[1*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        W[ka*n+1][col] = 1
        W[kb*n+1][col] = -1
        col += 1
        
        # M3
        U[0*m+ja][col] = 1
        V[ja*p+kb][col] = 1
        V[jb*p+kb][col] = -1
        W[kb*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M4
        U[1*m+jb][col] = 1
        V[jb*p+ka][col] = 1
        V[ja*p+ka][col] = -1
        W[ka*n+0][col] = 1
        W[ka*n+1][col] = 1
        col += 1
        
        # M5
        U[0*m+ja][col] = 1
        U[0*m+jb][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = -1
        W[kb*n+0][col] = 1
        col += 1
        
        # M6
        U[1*m+ja][col] = 1
        U[0*m+ja][col] = -1
        V[ja*p+ka][col] = 1
        V[ja*p+kb][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M7
        U[0*m+jb][col] = 1
        U[1*m+jb][col] = -1
        V[jb*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        col += 1

    # Blocks: partition J={0,1,2,3,4} and K={0,1,2,3,4}
    # Use Strassen for 2x2 blocks where possible
    # J pairs: (0,1), (2,3), (4,4) - but (4,4) is degenerate
    # K pairs: (0,1), (2,3), (4,4)
    
    # Four 2x2 Strassen blocks:
    add_strassen(0, 1, 0, 1)
    add_strassen(2, 3, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(2, 3, 2, 3)
    
    # Remaining pairs involving index 4
    # J in {0,1,2,3}, K=4: 4 pairs * 2 i-values = 8 terms
    for j in [0, 1, 2, 3]:
        k = 4
        # Term i=0
        U[0*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+0][col] = 1
        col += 1
        # Term i=1
        U[1*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1
        
    # J=4, K in {0,1,2,3,4}: 5 pairs * 2 i-values = 10 terms
    for k in [0, 1, 2, 3, 4]:
        j = 4
        # Term i=0
        U[0*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+0][col] = 1
        col += 1
        # Term i=1
        U[1*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if not np.allclose(T_recon, T_target, atol=1e-9, rtol=0):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school), dtype=int)
        V_school = np.zeros((m*p, rank_school), dtype=int)
        W_school = np.zeros((p*n, rank_school), dtype=int)
        
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col_s] = 1
                    V_school[j*p+k, col_s] = 1
                    W_school[k*n+i, col_s] = 1
                    col_s += 1
        
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U, "V": V, "W": W}
