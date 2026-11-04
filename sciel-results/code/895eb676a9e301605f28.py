import numpy as np

DIMENSIONS = [4, 4, 8]

# Strassen matrices for 2x2
U2 = np.array([
    [1, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1]
])

V2 = np.array([
    [1, 1, 0, -1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1]
])

W2 = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [1, -1, 1, 0, 0, 1, 0]
])

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4
    rank_4x4 = 49
    rank_total = 2 * rank_4x4
    
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # Construct 4x4 decomposition
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for r in range(16):
        i = r // 4
        j = r % 4
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        
        # V rows are j, k
        # W rows are k, i
        
        for c in range(49):
            u1 = c // 7
            u2 = c % 7
            
            # U
            U4[r, c] = U2[2*i1 + j1, u1] * U2[2*i2 + j2, u2]
            
            # V: row index r corresponds to j, k
            # Here r is the same as U's row index structure?
            # U rows: i, j. V rows: j, k.
            # So for V, we need to map r to j, k.
            # But we are filling V4 which is for 4x4.
            # In 4x4, V rows are j, k.
            # My loop variable r is used for U.
            # I need separate loops or careful mapping.
            
            # Let's restart the filling logic to be clearer.
            pass
            
    # Refill properly
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # U4 rows: i, j
    for i in range(4):
        for j in range(4):
            r_u = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    U4[r_u, c] = U2[2*i1 + j1, u1] * U2[2*i2 + j2, u2]
                    
    # V4 rows: j, k
    for j in range(4):
        for k in range(4):
            r_v = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    V4[r_v, c] = V2[2*j1 + k1, u1] * V2[2*j2 + k2, u2]
                    
    # W4 rows: k, i
    for k in range(4):
        for i in range(4):
            r_w = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            for u1 in range(7):
                for u2 in range(7):
                    c = u1 * 7 + u2
                    W4[r_w, c] = W2[2*k1 + i1, u1] * W2[2*k2 + i2, u2]
                    
    # Assemble full U, V, W
    # Block 1 (cols 0-48)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Block 2 (cols 49-97)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
