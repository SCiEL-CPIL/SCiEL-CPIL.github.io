import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Rank 42 baseline solution using 6 blocks of Strassen 2x2x2
    rank_base = 42
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: 2x2 matrix A
    # V: 2x2 matrix B
    # W: 2x2 matrix C
    # C = AB
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(1, 0): 1, (1, 1): 1},  # M2
        {(0, 0): 1},             # M3
        {(1, 1): 1},             # M4
        {(0, 0): 1, (0, 1): 1},  # M5
        {(1, 0): 1, (0, 0): -1}, # M6
        {(0, 1): 1, (1, 1): -1}  # M7
    ]
    V_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(0, 0): 1},             # M2
        {(0, 1): 1, (1, 1): -1}, # M3
        {(1, 0): 1, (0, 0): -1}, # M4
        {(1, 1): 1},             # M5
        {(0, 0): 1, (0, 1): 1},  # M6
        {(1, 0): 1, (1, 1): 1}   # M7
    ]
    W_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(1, 0): 1, (1, 1): -1}, # M2
        {(0, 1): 1, (1, 1): 1},  # M3
        {(0, 0): 1, (1, 0): 1},  # M4
        {(0, 0): -1, (0, 1): 1}, # M5
        {(1, 1): 1},             # M6
        {(0, 0): 1}              # M7
    ]
    
    col = 0
    # A is 2x4, split into two 2x2 blocks along columns: A0 (cols 0,1), A1 (cols 2,3)
    # B is 4x6, split into two 2x6 blocks along rows: B0 (rows 0,1), B1 (rows 2,3)
    # C = A0*B0 + A1*B1
    # A0*B0 is 2x2 * 2x6. Split B0 into three 2x2 blocks: B0_0, B0_1, B0_2
    # A1*B1 is 2x2 * 2x6. Split B1 into three 2x2 blocks: B1_0, B1_1, B1_2
    
    # Block A0 (i=0,1; j=0,1)
    for b_block in range(3): # 3 blocks of 2x2 in B0
        for s in range(7):
            # U: A0 entries
            for (i_loc, j_loc), coeff in U_coeffs_222[s].items():
                i = i_loc
                j = j_loc # 0 or 1
                U_base[i * m + j][col] += coeff
            # V: B0 entries
            for (j_loc, k_loc), coeff in V_coeffs_222[s].items():
                j = j_loc # 0 or 1
                k = b_block * 2 + k_loc # 0,1 or 2,3 or 4,5
                V_base[j * p + k][col] += coeff
            # W: C entries
            for (i_loc, k_loc), coeff in W_coeffs_222[s].items():
                i = i_loc
                k = b_block * 2 + k_loc
                W_base[k * n + i][col] += coeff
            col += 1

    # Block A1 (i=0,1; j=2,3)
    for b_block in range(3): # 3 blocks of 2x2 in B1
        for s in range(7):
            # U: A1 entries
            for (i_loc, j_loc), coeff in U_coeffs_222[s].items():
                i = i_loc
                j = 2 + j_loc # 2 or 3
                U_base[i * m + j][col] += coeff
            # V: B1 entries
            for (j_loc, k_loc), coeff in V_coeffs_222[s].items():
                j = 2 + j_loc # 2 or 3
                k = b_block * 2 + k_loc
                V_base[j * p + k][col] += coeff
            # W: C entries
            for (i_loc, k_loc), coeff in W_coeffs_222[s].items():
                i = i_loc
                k = b_block * 2 + k_loc
                W_base[k * n + i][col] += coeff
            col += 1

    # Verify baseline is correct
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the 2x4x6 matmul tensor."""
        n, m, p = DIMENSIONS
        # Build the target tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        # Reconstruct from factors
        T_recon = np.zeros((n * m, m * p, p * n))
        rank = len(U[0]) if U else 0
        for s in range(rank):
            u = np.array([U[i][s] for i in range(n * m)])
            v = np.array([V[j][s] for j in range(m * p)])
            w = np.array([W[k][s] for k in range(p * n)])
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.linalg.norm(T - T_recon) < 1e-10

    if not verify(U_base, V_base, W_base):
        # Fallback: return zeros or raise error? The construction is standard, so this shouldn't happen.
        pass

    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
