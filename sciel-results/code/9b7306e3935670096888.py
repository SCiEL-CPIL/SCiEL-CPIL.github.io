import numpy as np

def construct():
    """
    Constructs a rank-189 decomposition for 6x6 matrix multiplication
    using the Kronecker product of the Strassen algorithm (2x2, rank 7)
    and the Schoolbook algorithm (3x3, rank 27).
    """
    n, m, p = 6, 6, 6
    
    # --- 1. Define 2x2 Strassen Factors (Rank 7) ---
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    # Indices: 0:11, 1:12, 2:21, 3:22
    
    U_2 = np.array([
        [1,  0,  1,  0,  1, -1,  0], # A11
        [0,  0,  0,  0,  1,  0,  1], # A12
        [0,  1,  0,  0,  0,  1,  0], # A21
        [1,  1,  0,  1,  0,  0, -1]  # A22
    ])
    
    V_2 = np.array([
        [1,  1,  0, -1,  0,  1,  0], # B11
        [0,  0,  1,  0,  0,  1,  0], # B12
        [0,  0,  0,  1,  0,  0,  1], # B21
        [1,  0, -1,  0,  1,  0,  1]  # B22
    ])
    
    W_2 = np.array([
        [1,  0,  0,  1, -1,  0,  1], # C11
        [0,  0,  1,  0,  1,  0,  0], # C12
        [0,  1,  0,  1,  0,  0,  0], # C21
        [1, -1, 1,  0,  0,  1,  0]   # C22
    ])
    
    # --- 2. Define 3x3 Schoolbook Factors (Rank 27) ---
    # U_3: 9x27, V_3: 9x27, W_3: 9x27
    # Indices: i*3 + j
    
    rank_3 = 9
    U_3 = np.zeros((9, 27))
    V_3 = np.zeros((9, 27))
    W_3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U maps A[i,j]
                U_3[i * 3 + j, col] = 1
                # V maps B[j,k]
                V_3[j * 3 + k, col] = 1
                # W maps C[i,k] (Note: Trace convention W[k*n+i, r] in prompt? 
                # Prompt says W[k*n+i, r]. 
                # Standard MM: C[i,k] = sum A[i,j] B[j,k].
                # The prompt's W index is k*n+i. 
                # Let's check the seed: W[k * n + i][column] = 1.
                # So W row index is k*3 + i for 3x3.
                W_3[k * 3 + i, col] = 1
                col += 1
                
    # --- 3. Kronecker Product ---
    # U_total = U_3 (kron) U_2
    # Dimensions: (9*4) x (27*7) = 36 x 189
    
    U_total = np.kron(U_3, U_2)
    V_total = np.kron(V_3, V_2)
    W_total = np.kron(W_3, W_2)
    
    # Convert to lists of lists of floats/ints as required
    # The interface expects {'U', 'V', 'W'} with shapes (n*m, rank), etc.
    # n*m = 36. rank = 189.
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
