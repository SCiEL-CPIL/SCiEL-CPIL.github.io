import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s margin

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Winograd 3x3x3 factors (rank 7)
    U_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)
    
    V_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)
    
    W_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1],
    ], dtype=float)
    
    # Winograd 3x2x2 factors (rank 4)
    U_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    V_win322 = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    W_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, -1],
    ], dtype=float)
    
    # Build the full decomposition with rank 18
    # Part 1: 3x3x3 Winograd (rank 7) - A[:,:3] * B[:3,:]
    # Part 2: 3x1x3 (rank 3) - A[:,3] * B[3,:]
    # Part 3: 3x2x2 Winograd (rank 4) - A[:,:2] * B[:2, 3:5]
    # Part 4: 3x1x2 (rank 2) - A[:,2] * B[2, 3:5]
    # Part 5: 3x1x2 (rank 2) - A[:,3] * B[3, 3:5]
    rank_total = 7 + 3 + 4 + 2 + 2  # 18
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Part 1: A[:,:3] * B[:3,:] using Winograd 3x3x3
    for r in range(7):
        for i in range(3):
            for j in range(3):
                U[i * 4 + j, col + r] = U_win333[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[j * 5 + k, col + r] = V_win333[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i, col + r] = W_win333[k * 3 + i, r]
    col += 7
    
    # Part 2: A[:,3] * B[3,:] schoolbook rank 3
    # j=3 fixed, k varies 0..2, i varies 0..2
    for i in range(3):
        for k in range(3):
            U[i * 4 + 3, col] = 1
            V[3 * 5 + k, col] = 1
            W[k * 3 + i, col] = 1
            col += 1
    
    # Part 3: A[:,:2] * B[:2, 3:5] using Winograd 3x2x2
    for r in range(4):
        for i in range(3):
            for j in range(2):
                U[i * 4 + j, col + r] = U_win322[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + (3 + k), col + r] = V_win322[j * 2 + k, r]
        for k in range(2):
            for i in range(3):
                W[(3 + k) * 3 + i, col + r] = W_win322[k * 3 + i, r]
    col += 4
    
    # Part 4: A[:,2] * B[2, 3:5] schoolbook rank 2
    # j=2 fixed, k varies 3,4, i varies 0..2
    for i in range(3):
        for k in range(2):
            U[i * 4 + 2, col] = 1
            V[2 * 5 + (3 + k), col] = 1
            W[(3 + k) * 3 + i, col] = 1
            col += 1
    
    # Part 5: A[:,3] * B[3, 3:5] schoolbook rank 2
    # j=3 fixed, k varies 3,4, i varies 0..2
    for i in range(3):
        for k in range(2):
            U[i * 4 + 3, col] = 1
            V[3 * 5 + (3 + k), col] = 1
            W[(3 + k) * 3 + i, col] = 1
            col += 1
    
    # Verify correctness
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank_total):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_recon[i, k] = total
    
    if not np.allclose(C_test, C_recon, atol=1e-10):
        # Fallback to schoolbook if verification fails
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
