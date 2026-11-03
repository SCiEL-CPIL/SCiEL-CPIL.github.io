"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 Strassen multiplications
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 by 2x2
    # Local indices i, j, j', k in {0, 1}
    # U index: i*m + j (local m=2) -> global i*m + (2s+j)
    # V index: j'*p + k (local p=2) -> global (2s+j')*p + (2l+k)
    # W index: k*n + i (local n=2) -> global (2l+k)*n + i
    
    # Coefficients for the 7 multiplications
    # Each entry is a list of (local_index, coeff)
    # U_local: i, j
    # V_local: j', k
    # W_local: k, i
    
    strassen_steps = [
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)]
        },
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((0,1), 1), ((1,1), -1)] # C10 gets +1, C11 gets -1
        },
        {
            'U': [((0,0), 1)],
            'V': [((1,0), 1), ((1,1), -1)],
            'W': [((0,1), 1), ((1,1), 1)] # C01 gets +1, C11 gets +1
        },
        {
            'U': [((1,1), 1)],
            'V': [((0,0), 1), ((1,0), -1)],
            'W': [((0,0), 1), ((0,1), 1)] # C00 gets +1, C10 gets +1
        },
        {
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((0,0), -1), ((0,1), 1)] # C00 gets -1, C01 gets +1
        },
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((1,0), 1)],
            'W': [((1,1), 1)] # C11 gets +1
        },
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,1), 1), ((0,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)] # C00 gets +1, C11 gets +1
        }
    ]
    
    for s in range(2):
        for l in range(4):
            base_col = (s * 4 + l) * 7
            for t in range(7):
                col = base_col + t
                step = strassen_steps[t]
                
                for (li, lj), coeff in step['U']:
                    global_i = li
                    global_j = 2 * s + lj
                    U[global_i * m + global_j][col] += coeff
                    
                for (lj_prime, lk), coeff in step['V']:
                    global_j_prime = 2 * s + lj_prime
                    global_k = 2 * l + lk
                    V[global_j_prime * p + global_k][col] += coeff
                    
                for (lk, li), coeff in step['W']:
                    global_k = 2 * l + lk
                    global_i = li
                    W[global_k * n + global_i][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
