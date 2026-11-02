"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Split j (size m=4) into 2 blocks of size 2
    # Split k (size p=6) into 3 blocks of size 2
    # Each 2x2x2 block uses Strassen's algorithm (rank 7)
    # Total rank = 2 * 3 * 7 = 42
    
    num_s = m // 2
    num_t = p // 2
    rank = num_s * num_t * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2x2
    # Keys are (local_i, local_j) for U, (local_j, local_k) for V, (local_k, local_i) for W
    # Values are coefficients
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},       # M1: A00 + A11
        {(1,0): 1, (1,1): 1},       # M2: A10 + A11
        {(0,0): 1},                 # M3: A00
        {(1,1): 1},                 # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00 + A01
        {(1,0): 1, (0,0): -1},      # M6: A10 - A00
        {(0,1): 1, (1,1): -1}       # M7: A01 - A11
    ]
    
    strassen_V = [
        {(0,0): 1, (1,1): 1},       # M1: B00 + B11
        {(0,0): 1},                 # M2: B00
        {(1,0): 1, (1,1): -1},      # M3: B10 - B11
        {(0,1): 1, (0,0): -1},      # M4: B01 - B00
        {(1,1): 1},                 # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00 + B01
        {(1,0): 1, (1,1): 1}        # M7: B10 + B11
    ]
    
    strassen_W = [
        {(0,0): 1, (1,1): 1},       # M1: C00 + C11
        {(0,1): 1, (1,1): -1},      # M2: C10 - C11
        {(1,0): 1, (1,1): 1},       # M3: C01 + C11
        {(0,0): 1, (0,1): 1},       # M4: C00 + C10
        {(0,0): -1, (1,0): 1},      # M5: -C00 + C01
        {(1,1): 1},                 # M6: C11
        {(0,0): 1}                  # M7: C00
    ]
    
    col = 0
    for s in range(num_s):
        for t in range(num_t):
            for m_idx in range(7):
                # Map local indices to global
                # j' = 0 -> j = 2s, j' = 1 -> j = 2s+1
                # k' = 0 -> k = 2t, k' = 1 -> k = 2t+1
                # i' = 0 -> i = 0, i' = 1 -> i = 1
                
                for (i_loc, j_loc), val in strassen_U[m_idx].items():
                    i_glob = i_loc
                    j_glob = 2 * s + j_loc
                    U[i_glob * m + j_glob][col] += val
                    
                for (j_loc, k_loc), val in strassen_V[m_idx].items():
                    j_glob = 2 * s + j_loc
                    k_glob = 2 * t + k_loc
                    V[j_glob * p + k_glob][col] += val
                    
                for (k_loc, i_loc), val in strassen_W[m_idx].items():
                    k_glob = 2 * t + k_loc
                    i_glob = i_loc
                    W[k_glob * n + i_glob][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
