import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 2, 16

    # Strassen 2x2x2 core coefficients
    # U: linear combinations of A entries (indices 0..3 for A00,A01,A10,A11)
    # V: linear combinations of B entries (local indices 0..3 for B00,B01,B10,B11)
    # W: linear combinations producing C entries (local indices 0..3 for C00,C01,C10,C11)
    
    U_strassen = [
        [1, 0, 0, 1],   # M1: (A00 + A11)
        [0, 0, 1, 1],   # M2: (A10 + A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00 + A01)
        [1, 0, 1, -1],  # M6: (A10 - A00)
        [0, 1, 0, -1],  # M7: (A01 - A11)
    ]
    
    V_strassen = [
        [1, 0, 0, 1],   # M1: (B00 + B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01 - B11)
        [0, 0, 1, -1],  # M4: (B10 - B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00 + B01)
        [0, 0, 1, 1],   # M7: (B10 + B11)
    ]
    
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00 += M1, C11 += M1
        [0, 0, 1, -1],  # M2: C10 += M2, C11 -= M2
        [0, 1, 0, 1],   # M3: C01 += M3, C11 += M3
        [1, 0, 1, 0],   # M4: C00 += M4, C10 += M4
        [-1, 1, 0, 0],  # M5: C00 -= M5, C01 += M5
        [0, 0, 0, 1],   # M6: C11 += M6
        [1, 0, 0, 0],   # M7: C00 += M7
    ]

    num_blocks = p // 2  # 8 blocks of 2 columns each
    R = num_blocks * 7   # 56 total rank

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    for q in range(num_blocks):
        # B local-to-global mapping for block q (columns 2q, 2q+1)
        # B[j, k] -> j*p + k
        b_map = [
            0 * p + 2 * q,       # B[0, 2q]
            0 * p + (2 * q + 1), # B[0, 2q+1]
            1 * p + 2 * q,       # B[1, 2q]
            1 * p + (2 * q + 1), # B[1, 2q+1]
        ]
        
        # C local-to-global mapping for block q
        # C[i, k] -> k*n + i
        c_map = [
            (2 * q) * n + 0,       # C[0, 2q]
            (2 * q + 1) * n + 0,   # C[0, 2q+1]
            (2 * q) * n + 1,       # C[1, 2q]
            (2 * q + 1) * n + 1,   # C[1, 2q+1]
        ]

        for s in range(7):
            r = q * 7 + s
            
            for a_idx in range(4):
                U[a_idx, r] = U_strassen[s][a_idx]
            
            for v_idx in range(4):
                V[b_map[v_idx], r] = V_strassen[s][v_idx]
            
            for w_idx in range(4):
                W[c_map[w_idx], r] = W_strassen[s][w_idx]

    # Quick verification
    if time.time() < deadline:
        np.random.seed(0)
        A_t = np.random.randint(-3, 4, (n, m))
        B_t = np.random.randint(-3, 4, (m, p))
        C_t = A_t @ B_t
        
        Af = A_t.flatten()
        Bf = B_t.flatten()
        Cdecomp = W @ (U.T @ Af) * (V.T @ Bf)
        Cref = C_t.T.flatten()
        
        if not np.array_equal(Cdecomp, Cref):
            # Fallback: schoolbook rank n*m*p = 64
            R_fb = n * m * p
            U = np.zeros((n * m, R_fb), dtype=int)
            V = np.zeros((m * p, R_fb), dtype=int)
            W = np.zeros((p * n, R_fb), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
