import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # We use a rank-21 decomposition derived from the Kronecker product structure.
    # This corresponds to the "Block Strassen" method where we split B into three 2x2 blocks
    # and apply the rank-7 Strassen algorithm to each block multiplication.
    # This is a distinct construction from the trivial (rank 24) or other potential methods.
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # The 7 products in Strassen's algorithm:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A11 - A01) * (B10 + B11)
    
    # U vectors (for A)
    U_strassen = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # V vectors (for B)
    V_strassen = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    # W vectors (for C)
    # C00 = P1 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W is indexed by [k*n + i, r] where k is col of C, i is row of C.
    # C is 2x2 for each block.
    # So for a 2x2 C matrix:
    # Index 0: C[0,0] (i=0, k=0)
    # Index 1: C[1,0] (i=1, k=0)
    # Index 2: C[0,1] (i=0, k=1)
    # Index 3: C[1,1] (i=1, k=1)
    
    W_strassen = []
    for l in range(7):
        w = [0.0, 0.0, 0.0, 0.0]
        # C00 (idx 0)
        if l == 0: w[0] += 1
        if l == 4: w[0] -= 1
        if l == 6: w[0] += 1
        
        # C10 (idx 1)
        if l == 1: w[1] += 1
        if l == 3: w[1] += 1
        
        # C01 (idx 2)
        if l == 2: w[2] += 1
        if l == 4: w[2] += 1
        
        # C11 (idx 3)
        if l == 0: w[3] += 1
        if l == 1: w[3] -= 1
        if l == 2: w[3] += 1
        if l == 5: w[3] += 1
        
        W_strassen.append(w)
        
    # Now fill U, V, W for the 2x2x6 case
    for t in range(3):
        for l in range(7):
            k_global = t * 7 + l
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, k_global] = U_strassen[l][i * m + j]
            
            # Fill V
            for j in range(m):
                for k_local in range(2):
                    k_global_b = t * 2 + k_local
                    V[j * p + k_global_b, k_global] = V_strassen[l][j * 2 + k_local]
            
            # Fill W
            for i in range(n):
                for k_local in range(2):
                    k_global_c = t * 2 + k_local
                    W[k_global_c * n + i, k_global] = W_strassen[l][k_local * n + i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
