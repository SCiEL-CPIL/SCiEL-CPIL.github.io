"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    strassen_steps = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    # 8 blocks: 2 halves of A (L, R) x 4 column pairs of B
    for h in range(2):
        for c in range(4):
            for s in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_steps[s]
                
                # Map local 2x2 indices to global indices
                # A: i in {0,1}, j_local in {0,1} -> j_global = 2*h + j_local
                # B: j_local in {0,1}, k_local in {0,1} -> j_global = 2*h + j_local, k_global = 2*c + k_local
                # C: i in {0,1}, k_local in {0,1} -> k_global = 2*c + k_local
                
                # Fill U
                for local_idx, val in u_coeffs.items():
                    i = local_idx // 2
                    j_l = local_idx % 2
                    j_g = 2 * h + j_l
                    U[i * m + j_g][col] += val
                    
                # Fill V
                for local_idx, val in v_coeffs.items():
                    j_l = local_idx // 2
                    k_l = local_idx % 2
                    j_g = 2 * h + j_l
                    k_g = 2 * c + k_l
                    V[j_g * p + k_g][col] += val
                    
                # Fill W
                for local_idx, val in w_coeffs.items():
                    i = local_idx // 2
                    k_l = local_idx % 2
                    k_g = 2 * c + k_l
                    W[k_g * n + i][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
