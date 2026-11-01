"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r gives the tensor.
      Here n=m=p=4, so:
        U index: i*4 + k  (A row i, A col k)
        V index: k*4 + j  (B row k, B col j)
        W index: j*4 + i  (C col j, C row i) -- trace convention W[k*n+i, r]
    
    Strassen at the 4x4 level: split into 2x2 blocks of 2x2 matrices.
    Each 2x2 sub-multiplication uses Strassen (7 scalar mults).
    Total rank: 7 * 7 = 49.
    """
    t_start = time.time()
    deadline = 110.0  # safety margin before 150s kill

    n = m = p = 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen 2x2 formula:
    # Each entry: (A_coeffs, B_coeffs, C_contrib)
    # A_coeffs: dict {(r,c): coeff} for the 2x2 A matrix
    # B_coeffs: dict {(r,c): coeff} for the 2x2 B matrix
    # C_contrib: dict {(r,c): coeff} for the 2x2 C output

    strassen_2x2 = [
        # M1 = (a+d)(e+h) -> contributes to C[0,0], C[1,1]
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (c+d)*e -> contributes to C[1,0], C[1,1]
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a*(f-h) -> contributes to C[0,1], C[1,1]
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = d*(g-e) -> contributes to C[0,0], C[1,0]
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a+b)*h -> contributes to C[0,0], C[0,1]
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (c-a)*(e+f) -> contributes to C[1,1]
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (b-d)*(g+h) -> contributes to C[0,0]
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Block-level Strassen for 4x4 (same structure as 2x2 Strassen)
    block_strassen = strassen_2x2  # Same 7-term structure

    # For each block-level term t (0..6) and each scalar-level term s (0..6),
    # we get one rank-1 term with index r = t*7 + s

    for t in range(7):
        block_A, block_B, block_C = block_strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_2x2[s]
            r_idx = t * 7 + s

            if time.time() - t_start > deadline:
                return {"U": U, "V": V, "W": W}

            # Build U contribution
            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    u_idx = a_row * m + a_col
                    U[u_idx][r_idx] += block_coeff * scalar_coeff

            # Build V contribution
            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    v_idx = b_row * p + b_col
                    V[v_idx][r_idx] += block_coeff * scalar_coeff

            # Build W contribution
            # W uses trace convention: W[k*n+i, r] where i = row of C, k = col of C
            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr   # i index of C
                    c_col = 2 * cj + sc   # k index of C (column of C)
                    w_idx = c_col * n + c_row  # W[k*n + i]
                    W[w_idx][r_idx] += block_coeff * scalar_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
