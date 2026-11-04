import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
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
    
    # 2. Define 3x3 Schoolbook factors
    # U3: 9x27, V3: 9x27, W3: 9x27
    rank3 = 27
    U3 = np.zeros((9, rank3))
    V3 = np.zeros((9, rank3))
    W3 = np.zeros((9, rank3))
    
    col = 0
    for i0 in range(3):
        for j0 in range(3):
            for k0 in range(3):
                U3[i0 * 3 + j0, col] = 1
                V3[j0 * 3 + k0, col] = 1
                W3[k0 * 3 + i0, col] = 1
                col += 1
                
    # 3. Kronecker Product
    # U_full_kr: 36x189
    U_full_kr = np.kron(U2, U3)
    V_full_kr = np.kron(V2, V3)
    W_full_kr = np.kron(W2, W3)
    
    # 4. Permute rows to match required index ordering
    # Required index: i*6 + j = (3*i1 + i0)*6 + (3*j1 + j0) = 18*i1 + 6*i0 + 3*j1 + j0
    # Kron index: (2*i1 + j1)*9 + (3*i0 + j0) = 18*i1 + 9*j1 + 3*i0 + j0
    
    # We need to map from Kron index to Required index
    # Let perm_U[r_kr] = r_req
    # Then U_perm[r_req] = U_kr[r_kr]
    
    perm_U = np.zeros(36, dtype=int)
    perm_V = np.zeros(36, dtype=int)
    perm_W = np.zeros(36, dtype=int)
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(3):
                for j0 in range(3):
                    r_kr = (2 * i1 + j1) * 9 + (3 * i0 + j0)
                    r_req = 18 * i1 + 6 * i0 + 3 * j1 + j0
                    perm_U[r_kr] = r_req
                    
    # For V: index j*6 + k = (3*j1 + j0)*6 + (3*k1 + k0) = 18*j1 + 6*j0 + 3*k1 + k0
    # Kron index: (2*j1 + k1)*9 + (3*j0 + k0) = 18*j1 + 9*k1 + 3*j0 + k0
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(3):
                for k0 in range(3):
                    r_kr = (2 * j1 + k1) * 9 + (3 * j0 + k0)
                    r_req = 18 * j1 + 6 * j0 + 3 * k1 + k0
                    perm_V[r_kr] = r_req
                    
    # For W: index k*6 + i = (3*k1 + k0)*6 + (3*i1 + i0) = 18*k1 + 6*k0 + 3*i1 + i0
    # Kron index: (2*k1 + i1)*9 + (3*k0 + i0) = 18*k1 + 9*i1 + 3*k0 + i0
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(3):
                for i0 in range(3):
                    r_kr = (2 * k1 + i1) * 9 + (3 * k0 + i0)
                    r_req = 18 * k1 + 6 * k0 + 3 * i1 + i0
                    perm_W[r_kr] = r_req
                    
    U_final = np.zeros_like(U_full_kr)
    V_final = np.zeros_like(V_full_kr)
    W_final = np.zeros_like(W_full_kr)
    
    U_final[perm_U] = U_full_kr
    V_final[perm_V] = V_full_kr
    W_final[perm_W] = W_full_kr
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
