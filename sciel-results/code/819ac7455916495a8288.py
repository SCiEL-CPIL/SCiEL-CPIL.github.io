"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    t0 = time.time()
    
    # Use Rank 189 decomposition:
    # Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use schoolbook for 3x3 block multiplication (27 products).
    # Use Strassen (Rank 7) for each 2x2 block multiplication.
    # Total Rank = 27 * 7 = 189.
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen terms: (U_offsets, U_coeffs, V_offsets, V_coeffs)
    # Offsets: 0=A11, 1=A12, 2=A21, 3=A22
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions: list of (c_offset, coeff) for each term
    # C offsets: 0=C11, 1=C12, 2=C21, 3=C22
    c_contributions = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, -1)],  # M2
        [(1, 1), (3, 1)],   # M3
        [(0, 1), (2, 1)],   # M4
        [(0, -1), (1, 1)],  # M5
        [(3, 1)],           # M6
        [(0, 1)],           # M7
    ]
    
    col = 0
    # bi: block row of A (0..2)
    # bk: block col of A / block row of B (0..2)
    # bj: block col of B (0..2)
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Fill U
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bi + r
                        gc = 2 * bk + c
                        U[gr * 6 + gc][col] += coeff
                    
                    # Fill V
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bk + r
                        gc = 2 * bj + c
                        V[gr * 6 + gc][col] += coeff
                    
                    # Fill W
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        gr = 2 * bj + c
                        gc = 2 * bi + r
                        W[gr * 6 + gc][col] += coeff
                    
                    col += 1
    
    elapsed = time.time() - t0
    if elapsed > 140:
        # Safety margin before 150s kill
        pass  # We're already done constructing, just return
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
