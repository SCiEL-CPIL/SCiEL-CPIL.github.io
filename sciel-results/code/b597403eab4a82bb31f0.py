"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
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
    
    # A offsets: 0=(0,0)=A11, 1=(0,1)=A12, 2=(1,0)=A21, 3=(1,1)=A22
    # B offsets: 0=(0,0)=B11, 1=(0,1)=B12, 2=(1,0)=B21, 3=(1,1)=B22
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        # M2 = (A21+A22)B11
        (([2, 3], [1, 1]), ([0], [1])),
        # M3 = A11(B12-B22)
        (([0], [1]), ([1, 3], [1, -1])),
        # M4 = A22(B21-B11)
        (([3], [1]), ([2, 0], [1, -1])),
        # M5 = (A11+A12)B22
        (([0, 1], [1, 1]), ([3], [1])),
        # M6 = (A21-A11)(B11+B12)
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        # M7 = (A12-A22)(B21+B22)
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions for each Strassen term:
    # C offsets: 0=(0,0)=C11, 1=(0,1)=C12, 2=(1,0)=C21, 3=(1,1)=C22
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
    for bi in range(3):  # block row of A
        for bk in range(3):  # block col of A = block row of B
            for bj in range(3):  # block col of B
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Set U factors
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    # Set V factors
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    # Set W factors
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
