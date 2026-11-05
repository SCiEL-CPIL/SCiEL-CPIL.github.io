"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition: Strassen (7) * Schoolbook 3x3 (27)
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 block multiplication of 3x3 blocks
    # Each entry is (alpha_dict, beta_dict, gamma_dict)
    # Keys are block indices: '11', '12', '21', '22'
    strassen_defs = [
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}, {'11': 1, '22': 1}), # M1
        ({'21': 1, '22': 1}, {'11': 1}, {'21': 1, '22': -1}), # M2
        ({'11': 1}, {'12': 1, '22': -1}, {'12': 1, '22': 1}), # M3
        ({'22': 1}, {'21': 1, '11': -1}, {'11': 1, '21': 1}), # M4
        ({'11': 1, '12': 1}, {'22': 1}, {'11': -1, '12': 1}), # M5
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}, {'22': 1}), # M6
        ({'12': 1, '22': -1}, {'21': 1, '22': 1}, {'11': 1}) # M7
    ]
    
    def get_global_indices(block, i, j):
        # i, j are local indices 0..2
        if block == '11': return i, j
        if block == '12': return i, j + 3
        if block == '21': return i + 3, j
        if block == '22': return i + 3, j + 3
        raise ValueError("Invalid block")
        
    for s_idx, (alpha, beta, gamma) in enumerate(strassen_defs):
        base_col = s_idx * 27
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r_local = i * 9 + j * 3 + k
                    R = base_col + r_local
                    
                    # U contributions (A matrix)
                    for b, coeff_a in alpha.items():
                        if coeff_a != 0:
                            I, J = get_global_indices(b, i, j)
                            U[I * m + J][R] += coeff_a
                            
                    # V contributions (B matrix)
                    for b, coeff_b in beta.items():
                        if coeff_b != 0:
                            J, K = get_global_indices(b, j, k)
                            V[J * p + K][R] += coeff_b
                            
                    # W contributions (C matrix)
                    for b, coeff_c in gamma.items():
                        if coeff_c != 0:
                            K, I = get_global_indices(b, k, i)
                            W[K * n + I][R] += coeff_c
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
