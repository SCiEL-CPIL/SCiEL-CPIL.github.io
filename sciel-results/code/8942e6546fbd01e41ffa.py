"""Matrix multiplication 6x6x6 - Strassen block decomposition (3x3 blocks of 2x2)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    Schoolbook at the 3x3 block level (27 block multiplications),
    Strassen (rank 7) for each 2x2 block multiplication.
    Total rank = 27 * 7 = 189.
    
    W follows convention W[k*n+i, r] where (k,i) is (col_C, row_C).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    bs = 2       # block size
    nb = 3       # number of blocks per dimension
    
    total_rank = 27 * 7  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A * B = [[C11, C12], [C21, C22]]
    #
    # 7 Strassen terms:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 - p6
    
    strassen_A = [
        [1, 0, 0, 1],   # p1: a11+a22
        [0, 0, 1, 1],   # p2: a21+a22
        [1, 0, 0, 0],   # p3: a11
        [0, 0, 0, 1],   # p4: a22
        [1, 1, 0, 0],   # p5: a11+a12
        [-1, 0, 1, 0],  # p6: a21-a11
        [0, 1, 0, -1],  # p7: a12-a22
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # p1: b11+b22
        [1, 0, 0, 0],   # p2: b11
        [0, 1, 0, -1],  # p3: b12-b22
        [0, -1, 1, 0],  # p4: b21-b11
        [0, 0, 0, 1],   # p5: b22
        [1, 1, 0, 0],   # p6: b11+b12
        [0, 0, 1, 1],   # p7: b21+b22
    ]
    
    # Output coefficients: output_coeffs[c_local][t] where
    # c_local: 0=C11, 1=C12, 2=C21, 3=C22
    # t: 0..6 for p1..p7
    output_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 = p1 + p4 - p5 + p7
        [0, 0, 1, 0, 1, 0, 0],    # C12 = p3 + p5
        [0, 1, 0, 1, 0, 0, 0],    # C21 = p2 + p4
        [1, -1, 1, 0, 0, -1, 0],  # C22 = p1 - p2 + p3 - p6
    ]
    
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for t in range(7):
                    # U: A block at position (i, k), local entry
                    for r in range(bs):
                        for c in range(bs):
                            a_local = r * bs + c
                            a_coeff = strassen_A[t][a_local]
                            if a_coeff != 0:
                                u_idx = (i * bs + r) * m + (k * bs + c)
                                U[u_idx][col] = a_coeff
                    
                    # V: B block at position (k, j), local entry
                    for r in range(bs):
                        for c in range(bs):
                            b_local = r * bs + c
                            b_coeff = strassen_B[t][b_local]
                            if b_coeff != 0:
                                v_idx = (k * bs + r) * p + (j * bs + c)
                                V[v_idx][col] = b_coeff
                    
                    # W: C block at position (i, j), local entry
                    for r in range(bs):
                        for c in range(bs):
                            c_local = r * bs + c
                            c_coeff = output_coeffs[c_local][t]
                            if c_coeff != 0:
                                w_idx = (j * bs + c) * n + (i * bs + r)
                                W[w_idx][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
