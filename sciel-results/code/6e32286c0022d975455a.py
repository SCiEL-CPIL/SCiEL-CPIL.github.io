"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42: 2 blocks of A (2x2) x 3 blocks of B cols (2x2) x 7 Strassen terms
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (first matrix M)
    # Indices in M: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> i*2 + j_local
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    # Strassen coefficients for V (second matrix N)
    # Indices in N: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> j_local*2 + k_local
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    # Strassen coefficients for W (output R)
    # Indices in R: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> i*2 + k_local
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},  # s=0 (P1)
        {(1, 0): 1, (1, 1): -1}, # s=1 (P2)
        {(0, 1): 1, (1, 1): 1},  # s=2 (P3)
        {(0, 0): 1, (1, 0): 1},  # s=3 (P4)
        {(0, 0): -1, (0, 1): 1}, # s=4 (P5)
        {(1, 1): 1},             # s=5 (P6)
        {(0, 0): 1}              # s=6 (P7)
    ]

    col = 0
    for a in range(2): # A block: 0 for j in {0,1}, 1 for j in {2,3}
        for b in range(3): # B col block: 0 for k in {0,1}, 1 for {2,3}, 2 for {4,5}
            for s in range(7): # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
