import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    A_u = [
        {0:1, 3:1},
        {2:1, 3:1},
        {0:1},
        {3:1},
        {0:1, 1:1},
        {2:1, 0:-1},
        {1:1, 3:-1},
    ]
    
    W_w = [
        {0:1, 3:1},
        {1:1, 3:-1},
        {2:1, 3:1},
        {0:1, 1:-1},
        {0:-1, 2:1},
        {3:1},
        {0:1, 2:-1},
    ]
    
    for t in range(3):
        for s in range(7):
            col = t*7 + s
            for idx, coeff in A_u[s].items():
                U[idx][col] = coeff
            e_idx = 2*t
            f_idx = 2*t+1
            g_idx = 6+2*t
            h_idx = 6+2*t+1
            if s == 0:
                V[e_idx][col] = 1; V[h_idx][col] = 1
            elif s == 1:
                V[e_idx][col] = 1
            elif s == 2:
                V[g_idx][col] = 1; V[h_idx][col] = -1
            elif s == 3:
                V[f_idx][col] = 1; V[e_idx][col] = -1
            elif s == 4:
                V[h_idx][col] = 1
            elif s == 5:
                V[e_idx][col] = 1; V[g_idx][col] = 1
            elif s == 6:
                V[f_idx][col] = 1; V[h_idx][col] = 1
            for idx, coeff in W_w[s].items():
                W[idx][col] = coeff
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    # Fallback to schoolbook
                    rank_sb = n*m*p
                    U_sb = [[0]*rank_sb for _ in range(n*m)]
                    V_sb = [[0]*rank_sb for _ in range(m*p)]
                    W_sb = [[0]*rank_sb for _ in range(p*n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_sb[i2*m+j2][col] = 1
                                V_sb[j2*p+k2][col] = 1
                                W_sb[k2*n+i2][col] = 1
                                col += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
