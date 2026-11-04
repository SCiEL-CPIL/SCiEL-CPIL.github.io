import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 Strassen Tensors (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i,j], V[j,k], W[i,k] for i,j,k in {0,1}
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1
    
    # M3 = A00(B01-B11) -> C01 += M3
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 -= M4, C10 -= M4
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = -1; W_222[2, 3] = -1
    
    # M5 = (A00+A01)B11 -> C00 += M5, C01 -= M5
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C11 -= M7
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[3, 6] = -1

    # 2. Expand to 4x4x4 using Kronecker Product
    # U_444: 16x49, V_444: 16x49, W_444: 16x49
    U_444 = np.kron(U_222, U_222)
    V_444 = np.kron(V_222, V_222)
    W_444 = np.kron(W_222, W_222)
    
    # The Kronecker product np.kron(A, B) for 2x2 matrices results in a 4x4 matrix.
    # Row index r in 4x4 corresponds to (r//2, r%2) in the 2x2 blocks.
    # This matches the standard row-major index i = i1*2 + i2 for 4x4 matrices.
    # So U_444, V_444, W_444 are correctly aligned for 4x4x4 multiplication.
    
    # 3. Construct 4x4x3 Schoolbook Tensors (Rank 48)
    # We need to multiply A(4,4) by B(4,3) to get C(4,3).
    # Indices: i in 0..3, j in 0..3, k in 0..2 (local).
    # Global k for this part will be 4, 5, 6.
    U_443 = np.zeros((16, 48), dtype=int)
    V_443 = np.zeros((12, 48), dtype=int) # 4*3 = 12
    W_443 = np.zeros((12, 48), dtype=int) # 4*3 = 12
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                # U index: i*4 + j
                # V index: j*3 + k_local
                # W index: i*3 + k_local
                U_443[i*4 + j, col] = 1
                V_443[j*3 + k_local, col] = 1
                W_443[i*3 + k_local, col] = 1
                col += 1

    # 4. Combine
    # Total Rank = 49 + 48 = 97
    # U: 16x97
    # V: 16x97 (First 49 cols from 4x4x4, next 48 from 4x4x3)
    #     Wait, V for 4x4x4 is 16x49. V for 4x4x3 is 12x48.
    #     We need to align V dimensions.
    #     V_444 is 16x49 (indices j*4+k, j in 0..3, k in 0..3).
    #     V_443 is 12x48 (indices j*3+k, j in 0..3, k in 0..2).
    #     The global B matrix is 4x7.
    #     V should have shape (4*7, Rank) = (28, Rank).
    #     V_444 covers B columns 0..3. Indices 0..15 in V_444 correspond to j*4+k.
    #     V_443 covers B columns 4..6. Indices 0..11 in V_443 correspond to j*3+k.
    #     We need to map V_443 indices to global B indices.
    #     Global B index for j, k_global (4..6) is j*7 + k_global.
    #     Local k_local = k_global - 4.
    #     So V_global[j*7 + k_global, col] = V_443[j*3 + k_local, col].
    
    V = np.zeros((28, 97), dtype=int)
    W = np.zeros((28, 97), dtype=int) # W global shape: (4*7, Rank) = (28, Rank)
    # W for 4x4x4: 16x49 (indices i*4+k, i in 0..3, k in 0..3).
    # W for 4x4x3: 12x48 (indices i*3+k, i in 0..3, k in 0..2).
    # Global W index for i, k_global is i*7 + k_global.
    
    # Part 1: 4x4x4 (k=0..3)
    # U_444 is 16x49.
    U = np.hstack([U_444, U_443])
    
    # Map V_444 to V
    for j in range(4):
        for k in range(4):
            V[j*7 + k, 0:49] = V_444[j*4 + k, 0:49]
            
    # Map W_444 to W
    for i in range(4):
        for k in range(4):
            W[i*7 + k, 0:49] = W_444[i*4 + k, 0:49]
            
    # Map V_443 to V
    for j in range(4):
        for k_local in range(3):
            k_global = k_local + 4
            V[j*7 + k_global, 49:97] = V_443[j*3 + k_local, 0:48]
            
    # Map W_443 to W
    for i in range(4):
        for k_local in range(3):
            k_global = k_local + 4
            W[i*7 + k_global, 49:97] = W_443[i*3 + k_local, 0:48]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
