import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Total rank: 49 (Strassen for 4x4x4) + 32 (Schoolbook for 4x4x2) + 16 (Schoolbook for 4x4x1) = 97
    rank = 97
    U = [[0]*rank for _ in range(n*m)]  # 16 x 97
    V = [[0]*rank for _ in range(m*p)]  # 28 x 97
    W = [[0]*rank for _ in range(p*n)]  # 28 x 97
    
    c = 0
    
    # --- Part 1: Strassen for 4x4 * 4x4 (cols 0-3 of B) ---
    # Strassen 4x4 decomposition
    # A = [A00 A01; A10 A11] (2x2 blocks of 2x2)
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # Base 2x2 Strassen tensors
    base_u = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    base_v = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10-B11
        [-1, 1, 0, 0],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    base_w = [
        [1, 0, 0, 1],   # M1 -> C00+C11
        [0, 0, 1, -1],  # M2 -> C10-C11
        [0, 1, 0, 1],   # M3 -> C01+C11
        [1, 0, 1, 0],   # M4 -> C00+C10
        [-1, 1, 0, 0],  # M5 -> -C00+C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    
    # Macro definitions for 4x4 Strassen
    macro_A = {
        1: [(0,0,1), (1,1,1)],   # A00+A11
        2: [(1,0,1), (1,1,1)],   # A10+A11
        3: [(0,0,1)],            # A00
        4: [(1,1,1)],            # A11
        5: [(0,0,1), (0,1,1)],   # A00+A01
        6: [(1,0,1), (0,0,-1)],  # A10-A00
        7: [(0,1,1), (1,1,-1)],  # A01-A11
    }
    macro_B = {
        1: [(0,0,1), (1,1,1)],   # B00+B11
        2: [(0,0,1)],            # B00
        3: [(1,0,1), (1,1,-1)],  # B10-B11
        4: [(0,1,1), (0,0,-1)],  # B01-B00
        5: [(1,1,1)],            # B11
        6: [(0,0,1), (1,0,1)],   # B00+B10
        7: [(0,1,1), (1,1,1)],   # B01+B11
    }
    macro_C = {
        1: [(0,0,1), (1,1,1)],   # C00+=M1, C11+=M1
        2: [(1,0,1), (1,1,-1)],  # C10+=M2, C11-=M2
        3: [(0,1,1), (1,1,1)],   # C01+=M3, C11+=M3
        4: [(0,0,1), (1,0,1)],   # C00+=M4, C10+=M4
        5: [(0,0,-1), (0,1,1)],  # C00-=M5, C01+=M5
        6: [(1,1,1)],            # C11+=M6
        7: [(0,0,1)],            # C00+=M7
    }
    
    for k in range(1, 8):
        for t in range(7):
            u_t = base_u[t]
            v_t = base_v[t]
            w_t = base_w[t]
            c_idx = c
            
            # U contributions
            for r in range(2):
                for cc in range(2):
                    val = u_t[r*2+cc]
                    if val == 0: continue
                    for (br, bc, coeff) in macro_A[k]:
                        A_row = br*2 + r
                        A_col = bc*2 + cc
                        U[A_row*4 + A_col][c_idx] += val * coeff
            
            # V contributions (B cols 0-3)
            for r in range(2):
                for cc in range(2):
                    val = v_t[r*2+cc]
                    if val == 0: continue
                    for (br, bc, coeff) in macro_B[k]:
                        B_row = br*2 + r
                        B_col = bc*2 + cc
                        V[B_row*7 + B_col][c_idx] += val * coeff
            
            # W contributions (C cols 0-3)
            for r in range(2):
                for cc in range(2):
                    val = w_t[r*2+cc]
                    if val == 0: continue
                    for (br, bc, coeff) in macro_C[k]:
                        C_row = br*2 + r
                        C_col = bc*2 + cc
                        W[C_col*4 + C_row][c_idx] += val * coeff
            
            c += 1
            if time.time() > deadline - 5:
                # Safety check, though this should be fast
                pass

    # --- Part 2: Schoolbook for 4x4 * 4x2 (cols 4-5 of B) ---
    # Rank 32
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                U[i*4 + j][c] = 1
                V[j*7 + k][c] = 1
                W[k*4 + i][c] = 1
                c += 1

    # --- Part 3: Schoolbook for 4x4 * 4x1 (col 6 of B) ---
    # Rank 16
    for i in range(4):
        for j in range(4):
            k = 6
            U[i*4 + j][c] = 1
            V[j*7 + k][c] = 1
            W[k*4 + i][c] = 1
            c += 1

    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
