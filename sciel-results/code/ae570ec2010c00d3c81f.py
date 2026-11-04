"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use Strassen (rank 7) for each 2x2 block multiplication.
    # Use schoolbook (rank 9) for the 3x3 block structure.
    # Total rank = 3*3*3*7 = 189
    
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
    # C11 = M1 + M4 - M5 + M7  -> W index 0 (k=0, i=0)
    # C12 = M3 + M5             -> W index 2 (k=1, i=0)
    # C21 = M2 + M4             -> W index 1 (k=0, i=1)
    # C22 = M1 - M2 + M3 + M6  -> W index 3 (k=1, i=1)
    
    c_contributions = [
        [(0, 1), (3, 1)],   # M1 -> C11, C22
        [(1, 1), (3, -1)],  # M2 -> C21, C22
        [(2, 1), (3, 1)],   # M3 -> C12, C22
        [(0, 1), (1, 1)],   # M4 -> C11, C21
        [(0, -1), (2, 1)],  # M5 -> C11, C12
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
                    
                    # Set U factors: U[i*m + k, col]
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    # Set V factors: V[k*p + j, col]
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    # Set W factors: W[j*n + i, col]
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
    
    # Verify the decomposition using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                s = 0.0
                for r in range(rank):
                    s += U_np[i * m + k, r] * V_np[k * p + j, r] * W_np[j * n + i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
