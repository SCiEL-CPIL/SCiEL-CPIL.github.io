"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: partition 6x6 into 3x3 blocks of 2x2 matrices, use schoolbook
    for the 3x3 block structure (27 block products) and rank-7 Strassen for
    each 2x2 block multiplication.
    
    Total rank = 27 * 7 = 189.
    """
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),   # M1
        (([2, 3], [1, 1]), ([0], [1])),           # M2
        (([0], [1]), ([1, 3], [1, -1])),          # M3
        (([3], [1]), ([2, 0], [1, -1])),          # M4
        (([0, 1], [1, 1]), ([3], [1])),           # M5
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),    # M6
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),    # M7
    ]
    
    c_contributions = [
        [(0, 1), (3, 1)],   # M1 -> C11, C22
        [(2, 1), (3, -1)],  # M2 -> C21, C22
        [(1, 1), (3, 1)],   # M3 -> C12, C22
        [(0, 1), (2, 1)],   # M4 -> C11, C21
        [(0, -1), (1, 1)],  # M5 -> C11, C12
        [(3, 1)],           # M6 -> C22
        [(0, 1)],           # M7 -> C11
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    if time.time() > deadline:
                        break
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
