import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition for 2x2x2 (Rank 8 schoolbook)
    # U1 (4x8), V1 (4x8), W1 (4x8)
    U1 = np.zeros((4, 8))
    V1 = np.zeros((4, 8))
    W1 = np.zeros((4, 8))
    
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U1[i*2 + j, col] = 1
                V1[j*2 + k, col] = 1
                W1[k*2 + i, col] = 1
                col += 1
                
    # Decomposition for 2x2x1 (Rank 4 schoolbook)
    # U2 (4x4), V2 (2x4), W2 (2x4)
    # n=2, m=2, p=1
    U2 = np.zeros((4, 4))
    V2 = np.zeros((2, 4))
    W2 = np.zeros((2, 4))
    
    col = 0
    for i in range(2):
        for j in range(2):
        # k is only 0
            U2[i*2 + j, col] = 1
            V2[j*1 + 0, col] = 1
            W2[0*2 + i, col] = 1
            col += 1

    # Kronecker Product
    # U = U1 (x) U2 -> (4*4) x (8*4) = 16 x 32
    # V = V1 (x) V2 -> (4*2) x (8*4) = 8 x 32
    # W = W1 (x) W2 -> (4*2) x (8*4) = 8 x 32
    
    U_kron = np.kron(U1, U2)
    V_kron = np.kron(V1, V2)
    W_kron = np.kron(W1, W2)
    
    # U_kron is 16x32. This matches n*m = 4*4 = 16.
    # V_kron is 8x32. We need m*p = 4*7 = 28.
    # W_kron is 8x32. We need p*n = 7*4 = 28.
    
    # Padding
    U_final = U_kron
    V_final = np.zeros((28, 32))
    W_final = np.zeros((28, 32))
    
    # The Kronecker product V1 (x) V2 corresponds to indices (j1, k1) and (j2, k2).
    # j = j1*2 + j2, k = k1*1 + k2?
    # Actually, the tensor product structure implies:
    # A (4x4) is A1 (2x2) (x) A2 (2x2).
    # B (4x7) is B1 (2x2) (x) B2 (2x1) padded?
    # No, B is 4x7. We are decomposing 4x4x2.
    # The "2" in p comes from p1=2, p2=1.
    # So the non-zero part of V is for k in {0,1}.
    # V_final rows correspond to j in {0..3}, k in {0..6}.
    # The Kronecker V has rows for (j1, j2) and (k1, k2) where k1 in {0,1}, k2=0.
    # So the rows are effectively for k in {0,1}.
    # We just copy V_kron into the first 8 rows?
    # Wait, the index mapping for V in Kronecker:
    # Row index in V_kron is (j1, j2) * (k1, k2).
    # j = j1*2 + j2.
    # k = k1*p2 + k2 = k1*1 + 0 = k1.
    # So k is 0 or 1.
    # The rows in V_final are indexed by j*p + k = j*7 + k.
    # This is NOT the same as the Kronecker row index.
    # We need to map the Kronecker indices to the final indices.
    
    # Let's do it manually to be safe.
    U_final = np.zeros((16, 32))
    V_final = np.zeros((28, 32))
    W_final = np.zeros((28, 32))
    
    # Iterate over the 32 columns
    for c in range(32):
        c1 = c // 4
        c2 = c % 4
        
        # U
        # U1 row: i1*2 + j1
        # U2 row: i2*2 + j2
        # Final U row: i*4 + j where i = i1*2 + i2, j = j1*2 + j2
        # U_kron[c] is U1[:, c1] (x) U2[:, c2]
        # The values are 1 if U1[i1,j1,c1]=1 and U2[i2,j2,c2]=1
        
        # It is easier to just use the kron result and permute/pad.
        # But the row permutation for V and W is tricky.
        
        pass

    # Let's stick to the manual construction of the 32 terms to avoid index errors.
    # Each term is a product of a term from M_2,2,2 and a term from M_2,2,1.
    # Term t1 in M_2,2,2: (i1, j1, k1)
    # Term t2 in M_2,2,1: (i2, j2, k2=0)
    # Combined term: i = i1*2 + i2, j = j1*2 + j2, k = k1*1 + 0 = k1.
    # Wait, p2=1, so k2=0. k = k1*1 + 0 = k1.
    # So k is 0 or 1.
    
    U_final = np.zeros((16, 32))
    V_final = np.zeros((28, 32))
    W_final = np.zeros((28, 32))
    
    col = 0
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        # k2 is always 0
                        i = i1 * 2 + i2
                        j = j1 * 2 + j2
                        k = k1
                        
                        U_final[i * 4 + j, col] = 1
                        V_final[j * 7 + k, col] = 1
                        W_final[k * 4 + i, col] = 1
                        col += 1
                        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
