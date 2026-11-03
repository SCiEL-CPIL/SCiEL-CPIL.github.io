import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Define the 7 Strassen coefficients for 2x2x2 (and by extension 4x4x4 blocks)
    # coeff_A[s][I][J], coeff_B[s][J][K], coeff_C[s][I][K]
    coeff_A = np.zeros((7, 2, 2))
    coeff_B = np.zeros((7, 2, 2))
    coeff_C = np.zeros((7, 2, 2))
    
    # s=0: (A00+A11)(B00+B11) -> C00, C11
    coeff_A[0, 0, 0] = 1; coeff_A[0, 1, 1] = 1
    coeff_B[0, 0, 0] = 1; coeff_B[0, 1, 1] = 1
    coeff_C[0, 0, 0] = 1; coeff_C[0, 1, 1] = 1
    
    # s=1: (A10+A11)B00 -> C10, -C11
    coeff_A[1, 1, 0] = 1; coeff_A[1, 1, 1] = 1
    coeff_B[1, 0, 0] = 1
    coeff_C[1, 1, 0] = 1; coeff_C[1, 1, 1] = -1
    
    # s=2: A00(B01-B11) -> C01, C11
    coeff_A[2, 0, 0] = 1
    coeff_B[2, 0, 1] = 1; coeff_B[2, 1, 1] = -1
    coeff_C[2, 0, 1] = 1; coeff_C[2, 1, 1] = 1
    
    # s=3: A11(B10-B00) -> C00
    coeff_A[3, 1, 1] = 1
    coeff_B[3, 1, 0] = 1; coeff_B[3, 0, 0] = -1
    coeff_C[3, 0, 0] = 1
    
    # s=4: (A00+A01)B11 -> C01, C10
    coeff_A[4, 0, 0] = 1; coeff_A[4, 0, 1] = 1
    coeff_B[4, 1, 1] = 1
    coeff_C[4, 0, 1] = 1; coeff_C[4, 1, 0] = 1
    
    # s=5: (A10-A00)(B00+B01) -> -C00, C11
    coeff_A[5, 1, 0] = 1; coeff_A[5, 0, 0] = -1
    coeff_B[5, 0, 0] = 1; coeff_B[5, 0, 1] = 1
    coeff_C[5, 0, 0] = -1; coeff_C[5, 1, 1] = 1
    
    # s=6: (A01-A11)(B10+B11) -> C00, -C11
    coeff_A[6, 0, 1] = 1; coeff_A[6, 1, 1] = -1
    coeff_B[6, 1, 0] = 1; coeff_B[6, 1, 1] = 1
    coeff_C[6, 0, 0] = 1; coeff_C[6, 1, 1] = -1
    
    rank_49 = 49
    U_49 = np.zeros((16, rank_49))
    V_49 = np.zeros((16, rank_49)) # First 4 cols of B
    W_49 = np.zeros((16, rank_49)) # First 4 cols of C
    
    for s in range(7):
        for u in range(7):
            t = s * 7 + u
            for I in range(2):
                for J in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            # Global A index: i = 2*I + i_prime, j = 2*J + j_prime
                            a_idx = i * 4 + j
                            # Coefficient is coeff_A[s][I][J] * coeff_A[u][i_prime][j_prime]
                            val = coeff_A[s, I, J] * coeff_A[u, i_prime, j_prime]
                            if val != 0:
                                U_49[a_idx, t] = val
                            
            for J in range(2):
                for K in range(2):
                    for j_prime in range(2):
                        for k_prime in range(2):
                            # Global B index (for first 4 cols): j = 2*J + j_prime, k = 2*K + k_prime
                            # V shape is (20, rank). Index is 5*j + k.
                            b_idx = j * 5 + k
                            val = coeff_B[s, J, K] * coeff_B[u, j_prime, k_prime]
                            if val != 0:
                                V_49[b_idx, t] = val
                                
            for I in range(2):
                for K in range(2):
                    for i_prime in range(2):
                        for k_prime in range(2):
                            # Global C index (for first 4 cols): i = 2*I + i_prime, k = 2*K + k_prime
                            # W shape is (20, rank). Index is 4*k + i.
                            l_idx = k * 4 + i
                            val = coeff_C[s, I, K] * coeff_C[u, k_prime, i_prime]
                            if val != 0:
                                W_49[l_idx, t] = val

    # Add 16 terms for the 5th column (k=4)
    rank_total = rank_49 + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U_49
    V[:16, :49] = V_49 # V_49 has 16 rows (4x4), but V has 20 rows.
    # Wait, V_49 was defined as (16, 49).
    # V global has 20 rows.
    # The first 4 columns of B correspond to rows 0..15 of V?
    # V index is 5*j + k.
    # For k=0..3, j=0..3.
    # Indices:
    # j=0, k=0..3 -> 0, 1, 2, 3
    # j=1, k=0..3 -> 5, 6, 7, 8
    # j=2, k=0..3 -> 10, 11, 12, 13
    # j=3, k=0..3 -> 15, 16, 17, 18
    # So V_49 rows must be mapped to these indices.
    
    V_global = np.zeros((20, 49))
    for j in range(4):
        for k in range(4):
            V_global[5*j + k, :] = V_49[4*j + k, :]
            
    V[:, :49] = V_global
    
    # W_49 has 16 rows.
    # W index is 4*k + i.
    # For k=0..3, i=0..3.
    # Indices:
    # k=0, i=0..3 -> 0, 1, 2, 3
    # k=1, i=0..3 -> 4, 5, 6, 7
    # k=2, i=0..3 -> 8, 9, 10, 11
    # k=3, i=0..3 -> 12, 13, 14, 15
    # So W_49 rows map directly to 0..15.
    W[:, :49] = W_49
    
    # Add 16 terms for k=4
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        c = 49 + idx
        U[i*4+j, c] = 1
        V[j*5+4, c] = 1
        W[4*4+i, c] = 1 # 16+i
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
