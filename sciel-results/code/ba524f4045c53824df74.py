import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank calculation:
    # 5 blocks of 2x2x2 Strassen (rank 7 each) = 35
    # 1 block of 2x1x10 Rank-1 = 1
    # Total = 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 30 x 36
    W = [[0] * rank for _ in range(p * n)]  # 20 x 36
    
    col = 0
    
    # Strassen part for j in {0, 1}
    # A indices:
    # A00 -> U[0]
    # A01 -> U[1]
    # A10 -> U[3]
    # A11 -> U[4]
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # B indices for j=0: V[k0], V[k1]
        # B indices for j=1: V[10+k0], V[10+k1]
        
        # M1: (A00 + A11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        # C[0,k0] += M1, C[1,k1] += M1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B[0,k0]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        # C[1,k0] += M2
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M3: A00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        # C[0,k1] += M3
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M4: A11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B[1,k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        # C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        # C[1,k1] += M6
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        # C[0,k0] += M7
        W[k0 * 2 + 0][col] += 1
        col += 1
        
    # Rank-1 part for j=2
    # A indices: A02 -> U[2], A12 -> U[5]
    # B indices: B[2,k] -> V[20+k]
    # C indices: C[i,k] -> W[k*2+i]
    
    # U[2][col] = 1, U[5][col] = 1
    U[2][col] = 1
    U[5][col] = 1
    
    for k in range(p):
        V[20 + k][col] = 1
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
        
    col += 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx] * V_np[v_idx], W_np[w_idx])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to trivial if something went wrong
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col_sb = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj][col_sb] = 1
                                V_sb[jj * p + kk][col_sb] = 1
                                W_sb[kk * n + ii][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
