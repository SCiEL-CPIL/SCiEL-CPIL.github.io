import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=float)
    V_triv = np.zeros((m * p, rank_triv), dtype=float)
    W_triv = np.zeros((p * n, rank_triv), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Block size 3: 2x2 blocks of 3x3 matrices
    bs = 3
    nb = n // bs  # 2 blocks per dimension

    # Rank-21 decomposition for 3x3 matrix multiplication (Bini et al. 1979)
    # Each step defines:
    #   A_coeffs: dict {(row, col): coeff} for the 3x3 A submatrix
    #   B_coeffs: dict {(row, col): coeff} for the 3x3 B submatrix
    #   C_coeffs: dict {(row, col): coeff} for the 3x3 C submatrix
    #
    # The 21 products M_s = (sum_a a_s * A_a) * (sum_b b_s * B_b)
    # contribute to C via: C_c = sum_s c_{s,c} * M_s

    # Bini's rank-21 scheme for 3x3 matrix multiplication
    # Using the standard formulation with entries indexed (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    
    # Let's use a known rank-21 scheme. I'll use the one from Bini et al.
    # The scheme has 21 terms. Let me define them:
    
    # Using a compact representation:
    # Each term: (A_pattern, B_pattern, C_pattern)
    # where each pattern is a dict mapping (r,c) to coefficient
    
    bini_21 = [
        # Term 0
        (
            {(0,0): 1},
            {(0,0): 1},
            {(0,0): 1}
        ),
        # Term 1
        (
            {(0,1): 1},
            {(1,0): 1},
            {(0,1): 1}
        ),
        # Term 2
        (
            {(0,2): 1},
            {(2,0): 1},
            {(0,2): 1}
        ),
        # Term 3
        (
            {(1,0): 1},
            {(0,1): 1},
            {(1,0): 1}
        ),
        # Term 4
        (
            {(1,1): 1},
            {(1,1): 1},
            {(1,1): 1}
        ),
        # Term 5
        (
            {(1,2): 1},
            {(2,1): 1},
            {(1,2): 1}
        ),
        # Term 6
        (
            {(2,0): 1},
            {(0,2): 1},
            {(2,0): 1}
        ),
        # Term 7
        (
            {(2,1): 1},
            {(1,2): 1},
            {(2,1): 1}
        ),
        # Term 8
        (
            {(2,2): 1},
            {(2,2): 1},
            {(2,2): 1}
        ),
        # Term 9
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # Term 10
        (
            {(0,1): 1, (1,0): 1},
            {(0,1): 1, (1,0): 1},
            {(0,1): 1, (1,0): 1}
        ),
        # Term 11
        (
            {(0,2): 1, (2,0): 1},
            {(0,2): 1, (2,0): 1},
            {(0,2): 1, (2,0): 1}
        ),
        # Term 12
        (
            {(1,2): 1, (2,1): 1},
            {(1,2): 1, (2,1): 1},
            {(1,2): 1, (2,1): 1}
        ),
        # Term 13
        (
            {(0,0): 1, (2,2): 1},
            {(0,0): 1, (2,2): 1},
            {(0,0): 1, (2,2): 1}
        ),
        # Term 14
        (
            {(0,1): 1, (2,1): 1},
            {(1,0): 1, (1,2): 1},
            {(0,1): 1, (2,1): 1}
        ),
        # Term 15
        (
            {(0,2): 1, (1,2): 1},
            {(0,0): 1, (2,0): 1},
            {(0,2): 1, (1,2): 1}
        ),
        # Term 16
        (
            {(1,0): 1, (2,0): 1},
            {(0,1): 1, (2,1): 1},
            {(1,0): 1, (2,0): 1}
        ),
        # Term 17
        (
            {(0,0): 1, (0,1): 1, (0,2): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1}
        ),
        # Term 18
        (
            {(1,0): 1, (1,1): 1, (1,2): 1},
            {(1,0): 1, (1,1): 1, (1,2): 1},
            {(1,0): 1, (1,1): 1, (1,2): 1}
        ),
        # Term 19
        (
            {(2,0): 1, (2,1): 1, (2,2): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1}
        ),
        # Term 20
        (
            {(0,0): 1, (1,0): 1, (2,0): 1},
            {(0,0): 1, (1,0): 1, (2,0): 1},
            {(0,0): 1, (1,0): 1, (2,0): 1}
        ),
    ]

    # This simple scheme above is NOT correct - it's just a guess.
    # Let me use a verified rank-21 scheme.
    
    # Actually, let me use a well-known rank-21 scheme from the literature.
    # I'll use the scheme from Bini et al. (1979) which is well-documented.
    
    # A cleaner approach: use the known rank-21 decomposition.
    # Let me define it properly using the standard notation.
    
    # The 21 terms for 3x3 matrix multiplication:
    # I'll use a verified scheme. Let me define each term as (A_dict, B_dict, C_dict)
    
    # Using a standard rank-21 scheme:
    bini_21_correct = [
        # 1
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # 2
        ({(0,1): 1}, {(0,1): 1}, {(0,1): 1}),
        # 3
        ({(0,2): 1}, {(0,2): 1}, {(0,2): 1}),
        # 4
        ({(1,0): 1}, {(1,0): 1}, {(1,0): 1}),
        # 5
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # 6
        ({(1,2): 1}, {(1,2): 1}, {(1,2): 1}),
        # 7
        ({(2,0): 1}, {(2,0): 1}, {(2,0): 1}),
        # 8
        ({(2,1): 1}, {(2,1): 1}, {(2,1): 1}),
        # 9
        ({(2,2): 1}, {(2,2): 1}, {(2,2): 1}),
        # 10
        ({(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}, {(0,0): 1, (1,1): 1, (2,2): 1}),
        # 11
        ({(0,1): 1, (1,2): 1, (2,0): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}, {(0,1): 1, (1,2): 1, (2,0): 1}),
        # 12
        ({(0,2): 1, (1,0): 1, (2,1): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}, {(0,2): 1, (1,0): 1, (2,1): 1}),
        # 13
        ({(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (0,1): 1, (0,2): 1}, {(0,0): 1, (0,1): 1, (0,2): 1}),
        # 14
        ({(1,0): 1, (1,1): 1, (1,2): 1}, {(1,0): 1, (1,1): 1, (1,2): 1}, {(1,0): 1, (1,1): 1, (1,2): 1}),
        # 15
        ({(2,0): 1, (2,1): 1, (2,2): 1}, {(2,0): 1, (2,1): 1, (2,2): 1}, {(2,0): 1, (2,1): 1, (2,2): 1}),
        # 16
        ({(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}, {(0,0): 1, (1,0): 1, (2,0): 1}),
        # 17
        ({(0,1): 1, (1,1): 1, (2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}, {(0,1): 1, (1,1): 1, (2,1): 1}),
        # 18
        ({(0,2): 1, (1,2): 1, (2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}, {(0,2): 1, (1,2): 1, (2,2): 1}),
        # 19
        ({(0,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 20
        ({(0,1): 1, (1,2): 1}, {(0,1): -1, (1,2): 1}, {(0,1): 1, (1,2): 1}),
        # 21
        ({(0,2): 1, (1,0): 1}, {(0,2): -1, (1,0): 1}, {(0,2): 1, (1,0): 1}),
    ]

    # This is still not verified. Let me take a different, safer approach.
    # I'll use the block-Strassen with bs=2 (rank 189) which is verified to work,
    # and also try bs=3 with Strassen (rank 7*27=189, same).
    # 
    # Actually, let me just use the verified bs=2 Block-Strassen (rank 189) as the main
    # solution, since the rank-21 scheme for 3x3 requires careful verification.
    
    # Use Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3

    strassen_steps = [
        {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1, (1, 1): 1}},
        {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1},
         'C': {(1, 0): 1, (1, 1): -1}},
        {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1},
         'C': {(0, 1): 1, (1, 1): 1}},
        {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1},
         'C': {(0, 0): 1, (1, 0): 1}},
        {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1},
         'C': {(0, 0): -1, (0, 1): 1}},
        {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1},
         'C': {(1, 1): 1}},
        {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1}},
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_steps[s]
                    for (ar, ac), coeff_a in s_def['A'].items():
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    for (br, bc), coeff_b in s_def['B'].items():
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    for (cr, cc), coeff_c in s_def['C'].items():
                        i_glob = i_b * bs + cr
                        j_glob = j_b * bs + cc
                        W2[j_glob * n + i_glob, r_idx] += coeff_c
                    r_idx += 1

    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if not np.allclose(T_recon, T, atol=1e-9):
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
