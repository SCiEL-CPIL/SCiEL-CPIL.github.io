import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2 correspond to A indices (00, 01, 10, 11) -> 0, 1, 2, 3
    # Rows of V2 correspond to B indices (00, 01, 10, 11) -> 0, 1, 2, 3
    # Rows of W2 correspond to C indices (00, 01, 10, 11) -> 0, 1, 2, 3
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    
    # Schoolbook 3x3 matrices
    # Rank 27
    # U3: 9x27, V3: 9x27, W3: 9x27
    # Indices:
    # U3 row i*3+j, col i*3+j+k ? No.
    # Term (i,j,k): A_ij B_jk -> C_ik
    # U index (i,j). V index (j,k). W index (k,i).
    # Col index c = i*9 + j*3 + k.
    
    rank3 = 27
    U3 = [[0]*rank3 for _ in range(9)]
    V3 = [[0]*rank3 for _ in range(9)]
    W3 = [[0]*rank3 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U3[i*3 + j][c] = 1
                V3[j*3 + k][c] = 1
                W3[k*3 + i][c] = 1
                
    # Kronecker Product Construction
    # Total Rank = 7 * 27 = 189
    total_rank = 7 * 27
    
    # U6 shape: 36 x 189
    # Row index r = I*3 + i' where I in 0..1, i' in 0..2
    # Col index C = c2*27 + c3 where c2 in 0..6, c3 in 0..26
    U6 = [[0]*total_rank for _ in range(36)]
    V6 = [[0]*total_rank for _ in range(36)]
    W6 = [[0]*total_rank for _ in range(36)]
    
    for I in range(2):
        for i_prime in range(3):
            r = I * 3 + i_prime
            for c2 in range(7):
                u2_val = U2[I][c2]
                if u2_val == 0: continue
                for c3 in range(27):
                    u3_val = U3[i_prime][c3]
                    if u3_val == 0: continue
                    C = c2 * 27 + c3
                    U6[r][C] = u2_val * u3_val
                    
    # V6 shape: 36 x 189
    # Row index: j_global * 6 + k_global
    # j_global = J * 3 + j_prime
    # k_global = K * 3 + k_prime
    # V2 row index: J * 2 + K
    # V3 row index: j_prime * 3 + k_prime
    for J in range(2):
        for j_prime in range(3):
            for K in range(2):
                for k_prime in range(3):
                    j_glob = J * 3 + j_prime
                    k_glob = K * 3 + k_prime
                    r_v = j_glob * 6 + k_glob
                    
                    v2_idx = J * 2 + K
                    v3_idx = j_prime * 3 + k_prime
                    
                    for c2 in range(7):
                        v2_val = V2[v2_idx][c2]
                        if v2_val == 0: continue
                        for c3 in range(27):
                            v3_val = V3[v3_idx][c3]
                            if v3_val == 0: continue
                            C = c2 * 27 + c3
                            V6[r_v][C] = v2_val * v3_val
                            
    # W6 shape: 36 x 189
    # Row index: k_global * 6 + i_global
    # k_global = K * 3 + k_prime
    # i_global = I * 3 + i_prime
    # W2 row index: K * 2 + I
    # W3 row index: k_prime * 3 + i_prime
    for K in range(2):
        for k_prime in range(3):
            for I in range(2):
                for i_prime in range(3):
                    k_glob = K * 3 + k_prime
                    i_glob = I * 3 + i_prime
                    r_w = k_glob * 6 + i_glob
                    
                    w2_idx = K * 2 + I
                    w3_idx = k_prime * 3 + i_prime
                    
                    for c2 in range(7):
                        w2_val = W2[w2_idx][c2]
                        if w2_val == 0: continue
                        for c3 in range(27):
                            w3_val = W3[w3_idx][c3]
                            if w3_val == 0: continue
                            C = c2 * 27 + c3
                            W6[r_w][C] = w2_val * w3_val
                            
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
