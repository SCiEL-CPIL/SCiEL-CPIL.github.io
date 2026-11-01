import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 blocks * 7 Strassen steps
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 matrix multiplication (rank 7)
    # Local indices: A/B/C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    v_vecs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    # W mapping: local 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    # Global W index: col_C * n + row_C
    w_vecs = [
        [1, 0, 0, 1],   # M1: c00 + c11
        [0, 0, 1, -1],  # M2: c10 - c11
        [0, 1, 0, 1],   # M3: c01 + c11
        [1, 0, 1, 0],   # M4: c00 + c10
        [-1, 1, 0, 0],  # M5: -c00 + c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V: B indices: li=0->B[0,2r]=V[2r], li=1->B[0,2r+1]=V[2r+1]
            #     li=2->B[1,2r]=V[16+2r], li=3->B[1,2r+1]=V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W: C indices: li=0->C[0,2r]=W[4r], li=1->C[0,2r+1]=W[4r+2]
            #     li=2->C[1,2r]=W[4r+1], li=3->C[1,2r+1]=W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
                    
    # Quick verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    valid = True
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for l in range(p):
                        val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+l, :] * W_arr[l*n+ip, :])
                        expected = 1.0 if (i == ip and j == jp) else 0.0
                        if abs(val - expected) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook rank 64
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+l][col] = 1
                    W_fb[l*n+i][col] = 1
                    col += 1
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
