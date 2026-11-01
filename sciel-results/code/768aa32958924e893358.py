import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 21
    
    # Initialize U, V, W as numpy arrays of zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen decomposition parameters for a 2x2 block
    # We apply this 3 times for the 2x6 case (blocks of 2 columns)
    
    # Indices for U (A matrix, 2x2)
    # U index: i * m + j. i in {0,1}, j in {0,1}
    # Values are coefficients for M1..M7
    
    # M1: A00 + A11 -> U[0][col]=1, U[3][col]=1
    # M2: A10 + A11 -> U[2][col]=1, U[3][col]=1
    # M3: A00       -> U[0][col]=1
    # M4: A11       -> U[3][col]=1
    # M5: A00 + A01 -> U[0][col]=1, U[1][col]=1
    # M6: A10 - A00 -> U[2][col]=1, U[0][col]=-1
    # M7: A01 - A11 -> U[1][col]=1, U[3][col]=-1
    
    U_updates = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]

    # Indices for V (B matrix, 2x2 local)
    # V index: j * p + k_global. j in {0,1}, k_global in {0..5}
    # For block r, k_global = 2*r + k_local. k_local in {0,1}
    
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # V_updates[k] is list of (j_local, k_local, coeff)
    V_updates = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]

    # Indices for W (C matrix, 2x2 local)
    # W index: k_global * n + i. k_global in {0..5}, i in {0,1}
    # For block r, k_global = 2*r + j_local. j_local in {0,1}
    
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # W_updates[k] is list of (i, j_local, coeff)
    # k=0 (M1): C00(1), C11(1)
    # k=1 (M2): C10(1), C11(-1)
    # k=2 (M3): C01(1), C11(1)
    # k=3 (M4): C00(1), C10(1)
    # k=4 (M5): C00(-1), C01(1)
    # k=5 (M6): C11(1)
    # k=6 (M7): C00(1)
    
    W_updates = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Construct the full rank-21 decomposition by iterating over 3 blocks
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            
            # Update U
            for idx, val in U_updates[k]:
                U[idx, col] = val
                
            # Update V
            for j_loc, k_loc, val in V_updates[k]:
                k_glob = 2 * r + k_loc
                v_idx = j_loc * p + k_glob
                V[v_idx, col] = val
                
            # Update W
            for i, j_loc, val in W_updates[k]:
                k_glob = 2 * r + j_loc
                w_idx = k_glob * n + i
                W[w_idx, col] = val
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Convert to lists for the required return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
