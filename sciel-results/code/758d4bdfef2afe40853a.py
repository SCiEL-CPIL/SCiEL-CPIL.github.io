"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size is 3
    bs = 3
    
    # Strassen coefficients for 2x2 block multiplication
    # alpha: coefficients for A blocks (11, 12, 21, 22)
    # beta: coefficients for B blocks (11, 12, 21, 22)
    strassen_terms = [
        ((1, 0, 0, 1), (1, 0, 0, 1)),  # M1
        ((0, 0, 1, 1), (1, 0, 0, 0)),  # M2
        ((1, 0, 0, 0), (0, 1, 0, -1)), # M3
        ((0, 0, 0, 1), (0, 0, 1, -1)), # M4
        ((1, 1, 0, 0), (0, 0, 0, 1)),  # M5
        ((-1, 0, 1, 0), (1, 1, 0, 0)), # M6
        ((0, 1, 0, -1), (0, 0, 1, 1))  # M7
    ]
    
    # Coefficients for combining M1..M7 into C blocks
    # gamma[block][term_index]
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M7
    gamma = [
        [1, 0, 0, 1, -1, 0, -1], # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 0, 0, 0, 1, 1]   # C22
    ]
    
    rank = 7 * (bs * bs * bs) # 7 * 27 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get row index for a block entry
    def get_row(block_id, i, j):
        # block_id: 0 for 11, 1 for 12, 2 for 21, 3 for 22
        # i, j are local indices 0..bs-1
        # Global row index in the 6x6 matrix
        # 11: 0..2, 0..2 -> 0..8
        # 12: 0..2, 3..5 -> 9..17
        # 21: 3..5, 0..2 -> 18..26
        # 22: 3..5, 3..5 -> 27..35
        return block_id * (bs * bs) + i * bs + j
    
    for k, (alpha, beta) in enumerate(strassen_terms):
        for i in range(bs):
            for j in range(bs):
                for l in range(bs):
                    # Column index for this scalar multiplication
                    # 27 multiplications per term
                    col = k * (bs * bs * bs) + i * (bs * bs) + j * bs + l
                    
                    # Update U: A entries
                    # X[i,j] = sum(alpha_b * A_b[i,j])
                    for b in range(4):
                        if alpha[b] != 0:
                            row = get_row(b, i, j)
                            U[row][col] = alpha[b]
                    
                    # Update V: B entries
                    # Y[j,l] = sum(beta_b * B_b[j,l])
                    for b in range(4):
                        if beta[b] != 0:
                            row = get_row(b, j, l)
                            V[row][col] = beta[b]
                    
                    # Update W: C entries
                    # Product contributes to M_k[i,l]
                    # M_k[i,l] contributes to C_b[i,l] with coeff gamma[b][k]
                    for b in range(4):
                        if gamma[b][k] != 0:
                            row = get_row(b, i, l)
                            W[row][col] = gamma[b][k]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
