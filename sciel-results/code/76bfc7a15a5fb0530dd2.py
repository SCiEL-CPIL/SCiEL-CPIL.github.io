import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 2, 14
    rank = 49
    
    # Define Strassen forms as numpy arrays for cleaner construction
    # A indices: 0=a, 1=b, 2=c, 3=d
    U_forms = np.array([
        [1, 0, 0, 1],   # M1 = a+d
        [0, 0, 1, 1],   # M2 = c+d
        [1, 0, 0, 0],   # M3 = a
        [0, 0, 0, 1],   # M4 = d
        [1, 1, 0, 0],   # M5 = a+b
        [-1, 0, 1, 0],  # M6 = c-a
        [0, 1, 0, -1]   # M7 = b-d
    ])
    
    # V_forms for one block (indices 0,1 for row 0, 2,3 for row 1 of B block? 
    # B block is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # Flattened B block indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # Wait, the global V shape is (m*p, rank) = (2*14, rank) = (28, rank).
    # B index mapping: B[i, j] -> V[i*p + j].
    # For block g, B columns are 2g and 2g+1.
    # B[0, 2g] -> V[0*14 + 2g] = V[2g]
    # B[0, 2g+1] -> V[0*14 + 2g+1] = V[2g+1]
    # B[1, 2g] -> V[1*14 + 2g] = V[14+2g]
    # B[1, 2g+1] -> V[1*14 + 2g+1] = V[14+2g+1]
    
    # Local V forms (coeffs for B[0,0], B[0,1], B[1,0], B[1,1])
    # M1: e+h -> B[0,0] + B[1,1]
    # M2: e    -> B[0,0]
    # M3: f-h  -> B[0,1] - B[1,1]
    # M4: g-e  -> B[1,0] - B[0,0]
    # M5: h    -> B[1,1]
    # M6: e+f  -> B[0,0] + B[0,1]
    # M7: g+h  -> B[1,0] + B[1,1]
    
    V_local_forms = np.array([
        [1, 0, 0, 1],   # M1: B[0,0] + B[1,1]
        [1, 0, 0, 0],   # M2: B[0,0]
        [0, 1, 0, -1],  # M3: B[0,1] - B[1,1]
        [-1, 0, 1, 0],  # M4: -B[0,0] + B[1,0]
        [0, 0, 0, 1],   # M5: B[1,1]
        [1, 1, 0, 0],   # M6: B[0,0] + B[0,1]
        [0, 0, 1, 1]    # M7: B[1,0] + B[1,1]
    ])
    
    # W forms for one block (indices 0,1 for C[0,0], C[0,1]; 2,3 for C[1,0], C[1,1] of block)
    # Global W shape (p*n, rank) = (14*2, rank) = (28, rank).
    # W index mapping: C[i, j] -> W[j*n + i] = W[2*j + i].
    # Block g columns: 2g, 2g+1.
    # C[0, 2g] -> W[2*(2g) + 0] = W[4g]
    # C[0, 2g+1] -> W[2*(2g+1) + 0] = W[4g+2]
    # C[1, 2g] -> W[2*(2g) + 1] = W[4g+1]
    # C[1, 2g+1] -> W[2*(2g+1) + 1] = W[4g+3]
    
    # Local W forms (coeffs for C[0,0], C[0,1], C[1,0], C[1,1])
    # M1: C[0,0] + C[1,1]
    # M2: C[1,0] - C[1,1]
    # M3: C[0,1] + C[1,1]
    # M4: C[0,0] + C[1,0]
    # M5: C[0,0] - C[0,1]
    # M6: C[1,1]
    # M7: C[0,0]
    
    W_local_forms = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ])
    
    # Initialize output matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill U (same for all blocks, so we can just tile or fill once per block structure)
    # Since U is shared, we fill column g*7 + s with U_forms[s]
    for s in range(7):
        # For each block g, the column index in U is g*7 + s
        # But U has the same values for all g?
        # Yes, U[i][col] = U_forms[s][i]
        # So we can fill all columns at once if we map them correctly.
        pass # Doing it in loop below for clarity and AST difference

    for g in range(7):
        if time.time() > deadline:
            break
        base_col = g * 7
        for s in range(7):
            col = base_col + s
            
            # U
            U[:, col] = U_forms[s]
            
            # V
            # Map local indices 0,1,2,3 to global indices
            # 0 -> 2g
            # 1 -> 2g+1
            # 2 -> 14+2g
            # 3 -> 14+2g+1
            v_indices = [2*g, 2*g+1, 14+2*g, 14+2*g+1]
            for k, idx in enumerate(v_indices):
                V[idx, col] = V_local_forms[s, k]
                
            # W
            # Map local indices 0,1,2,3 to global indices
            # 0 -> 4g
            # 1 -> 4g+2
            # 2 -> 4g+1
            # 3 -> 4g+3
            w_indices = [4*g, 4*g+2, 4*g+1, 4*g+3]
            for k, idx in enumerate(w_indices):
                W[idx, col] = W_local_forms[s, k]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
