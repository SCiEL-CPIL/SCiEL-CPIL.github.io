"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns the standard Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2 + j], V[j*2 + k], W[k*2 + i]
    """
    # The 7 terms for Strassen:
    # M1 = (A11 + A22)(B12 - B22)
    # M2 = (A11 + A12)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A22)B11
    # M6 = (A21 - A11)(B11 + B22)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M6 + M5
    # C22 = M1 - M3 + M4 + M6
    
    # U[i, t] corresponds to coefficient of A[i,j] in term t
    # V[j, t] corresponds to coefficient of B[j,k] in term t
    # W[k, t] corresponds to coefficient of C[i,k] in term t
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A11 + A22)(B12 - B22)
    U[0][0] = 1; U[3][0] = 1
    V[1][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2 = (A11 + A12)B22
    U[0][1] = 1; U[1][1] = 1
    V[3][1] = 1
    W[1][1] = 1
    
    # Term 2: M3 = A11(B11 - B22)
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # Term 3: M4 = A22(B11 - B12)
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[3][3] = 1
    
    # Term 4: M5 = (A11 + A22)B11
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5: M6 = (A21 - A11)(B11 + B22)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = -1
    
    return U, V, W

def _strassen_block(n, m, p):
    """
    Construct a block-Strassen decomposition.
    We tile the matrices with 2x2 blocks where possible.
    For 3x5*5x6:
    A is 3x5. We can view it as:
    Row 0: [A00 2x2 | A01 2x1 | A02 1x1] -- this is messy.
    
    Alternative approach:
    Decompose into blocks of size 2x2x2 and 1x1x1.
    
    Let's split indices:
    i in {0,1,2}
    j in {0,1,2,3,4}
    k in {0,1,2,3,4,5}
    
    We can group:
    i: {0,1}, {2}
    j: {0,1}, {2,3}, {4}
    k: {0,1}, {2,3}, {4,5}
    
    This gives a block structure:
    (2+1) x (2+2+1) x (2+2+2)
    
    Blocks:
    (2,2,2): Strassen rank 7
    (2,2,1): Schoolbook 2*2*1 = 4? Or can we do better?
              Actually, for 2x2 * 2x1 -> 2x1, rank is 4 (schoolbook) or maybe less?
              Tensor for 2x2 * 2x1:
              C11 = A11B11 + A12B21
              C12 = A11B12 + A12B22
              C21 = A21B11 + A22B21
              C22 = A21B12 + A22B22
              This is just two independent 2x1 * 2x1 multiplications? No.
              It's 2x2 matrix times 2x1 vector.
              Rank is 4. (No Strassen-like reduction for rectangular 2x2x1 known to be <4)
              
    (2,1,2): 2x1 * 1x2 -> 2x2. Rank 4.
    (1,2,2): 1x2 * 2x2 -> 1x2. Rank 4.
    (2,1,1): 2x1 * 1x1 -> 2x1. Rank 2.
    (1,2,1): 1x2 * 2x1 -> 1x1. Rank 2.
    (1,1,2): 1x1 * 1x2 -> 1x2. Rank 2.
    (2,2,2): Rank 7.
    
    Let's map the blocks:
    I1 = {0,1}, I2 = {2}
    J1 = {0,1}, J2 = {2,3}, J3 = {4}
    K1 = {0,1}, K2 = {2,3}, K3 = {4,5}
    
    Blocks (I, J, K):
    1. (2, 2, 2) -> Rank 7
    2. (2, 2, 2) -> Rank 7
    3. (2, 2, 2) -> Rank 7
    4. (2, 1, 2) -> Rank 4
    5. (2, 1, 2) -> Rank 4
    6. (2, 1, 2) -> Rank 4
    7. (1, 2, 2) -> Rank 4
    8. (1, 2, 2) -> Rank 4
    9. (1, 2, 2) -> Rank 4
    10. (1, 1, 2) -> Rank 2
    11. (1, 1, 2) -> Rank 2
    12. (1, 1, 2) -> Rank 2
    
    Total Rank = 3*7 + 3*4 + 3*4 + 3*2 = 21 + 12 + 12 + 6 = 51.
    
    Wait, is 51 correct?
    Schoolbook is 90.
    51 is much better.
    
    Let's double check the dimensions.
    A is 3x5.
    Rows: 0,1 (size 2), 2 (size 1).
    Cols: 0,1 (size 2), 2,3 (size 2), 4 (size 1).
    
    B is 5x6.
    Rows: 0,1 (size 2), 2,3 (size 2), 4 (size 1).
    Cols: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2).
    
    C is 3x6.
    Rows: 0,1 (size 2), 2 (size 1).
    Cols: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2).
    
    The blocks are products A_block * B_block.
    
    Block (I1, J1, K1): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]. Size 2x2x2. Rank 7.
    Block (I1, J1, K2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]. Size 2x2x2. Rank 7.
    Block (I1, J1, K3): A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]. Size 2x2x2. Rank 7.
    
    Block (I1, J2, K1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]. Size 2x2x2. Rank 7.
    Block (I1, J2, K2): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]. Size 2x2x2. Rank 7.
    Block (I1, J2, K3): A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]. Size 2x2x2. Rank 7.
    
    Block (I1, J3, K1): A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2]. Size 2x1x2. Rank 4.
    Block (I1, J3, K2): A[0:2, 4:5] * B[4:5, 2:4] -> C[0:2, 2:4]. Size 2x1x2. Rank 4.
    Block (I1, J3, K3): A[0:2, 4:5] * B[4:5, 4:6] -> C[0:2, 4:6]. Size 2x1x2. Rank 4.
    
    Block (I2, J1, K1): A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]. Size 1x2x2. Rank 4.
    Block (I2, J1, K2): A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]. Size 1x2x2. Rank 4.
    Block (I2, J1, K3): A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]. Size 1x2x2. Rank 4.
    
    Block (I2, J2, K1): A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]. Size 1x2x2. Rank 4.
    Block (I2, J2, K2): A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]. Size 1x2x2. Rank 4.
    Block (I2, J2, K3): A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]. Size 1x2x2. Rank 4.
    
    Block (I2, J3, K1): A[2:3, 4:5] * B[4:5, 0:2] -> C[2:3, 0:2]. Size 1x1x2. Rank 2.
    Block (I2, J3, K2): A[2:3, 4:5] * B[4:5, 2:4] -> C[2:3, 2:4]. Size 1x1x2. Rank 2.
    Block (I2, J3, K3): A[2:3, 4:5] * B[4:5, 4:6] -> C[2:3, 4:6]. Size 1x1x2. Rank 2.
    
    Total Rank = 6*7 (from I1,J1/K2/K3 and I1,J2/K1/K2/K3? No)
    Let's recount.
    I1 (size 2) blocks:
      J1 (size 2): K1, K2, K3 are size 2. So 3 blocks of 2x2x2. Rank 7 each. Total 21.
      J2 (size 2): K1, K2, K3 are size 2. So 3 blocks of 2x2x2. Rank 7 each. Total 21.
      J3 (size 1): K1, K2, K3 are size 2. So 3 blocks of 2x1x2. Rank 4 each. Total 12.
    
    I2 (size 1) blocks:
      J1 (size 2): K1, K2, K3 are size 2. So 3 blocks of 1x2x2. Rank 4 each. Total 12.
      J2 (size 2): K1, K2, K3 are size 2. So 3 blocks of 1x2x2. Rank 4 each. Total 12.
      J3 (size 1): K1, K2, K3 are size 2. So 3 blocks of 1x1x2. Rank 2 each. Total 6.
    
    Total Rank = 21 + 21 + 12 + 12 + 12 + 6 = 84.
    
    Wait, my previous count was 51. Why?
    Ah, I grouped J1 and J2 as "size 2" but there are 3 K blocks.
    I1 (2) x J1 (2) x K1 (2) -> 2x2x2
    I1 (2) x J1 (2) x K2 (2) -> 2x2x2
    I1 (2) x J1 (2) x K3 (2) -> 2x2x2
    That's 3 blocks of 2x2x2.
    
    I1 (2) x J2 (2) x K1 (2) -> 2x2x2
    I1 (2) x J2 (2) x K2 (2) -> 2x2x2
    I1 (2) x J2 (2) x K3 (2) -> 2x2x2
    That's 3 blocks of 2x2x2.
    
    I1 (2) x J3 (1) x K1 (2) -> 2x1x2
    I1 (2) x J3 (1) x K2 (2) -> 2x1x2
    I1 (2) x J3 (1) x K3 (2) -> 2x1x2
    That's 3 blocks of 2x1x2.
    
    I2 (1) x J1 (2) x K1 (2) -> 1x2x2
    I2 (1) x J1 (2) x K2 (2) -> 1x2x2
    I2 (1) x J1 (2) x K3 (2) -> 1x2x2
    That's 3 blocks of 1x2x2.
    
    I2 (1) x J2 (2) x K1 (2) -> 1x2x2
    I2 (1) x J2 (2) x K2 (2) -> 1x2x2
    I2 (1) x J2 (2) x K3 (2) -> 1x2x2
    That's 3 blocks of 1x2x2.
    
    I2 (1) x J3 (1) x K1 (2) -> 1x1x2
    I2 (1) x J3 (1) x K2 (2) -> 1x1x2
    I2 (1) x J3 (1) x K3 (2) -> 1x1x2
    That's 3 blocks of 1x1x2.
    
    Total Rank = 6*7 + 6*4 + 3*4 + 3*2 = 42 + 24 + 12 + 6 = 84.
    
    Okay, 84 is the rank.
    """
    n, m, p = 3, 5, 6
    
    # Define block ranges
    # I: [0,2), [2,3)
    # J: [0,2), [2,4), [4,5)
    # K: [0,2), [2,4), [4,6)
    
    I_ranges = [(0, 2), (2, 3)]
    J_ranges = [(0, 2), (2, 4), (4, 5)]
    K_ranges = [(0, 2), (2, 4), (4, 6)]
    
    U = [[0] * 84 for _ in range(n * m)]
    V = [[0] * 84 for _ in range(m * p)]
    W = [[0] * 84 for _ in range(p * n)]
    
    col_offset = 0
    
    for i_start, i_end in I_ranges:
        i_size = i_end - i_start
        for j_start, j_end in J_ranges:
            j_size = j_end - j_start
            for k_start, k_end in K_ranges:
                k_size = k_end - k_start
                
                # Determine block type and rank
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen
                    rank = 7
                    U_block, V_block, W_block = _strassen_2x2x2()
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    # 2x1x2 Schoolbook
                    rank = 4
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # A is 2x1, B is 1x2.
                    # C is 2x2.
                    # C[0,0] = A[0,0]B[0,0]
                    # C[0,1] = A[0,0]B[0,1]
                    # C[1,0] = A[1,0]B[0,0]
                    # C[1,1] = A[1,0]B[0,1]
                    U_block = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]] # Wait, U is (i*m+j)
                    # U index: i*m + j. Here m is local size? No, U is global.
                    # We need to map local indices to global indices.
                    
                    # Let's build the block matrices locally then place them.
                    # Local U: (i_size * j_size, rank)
                    # Local V: (j_size * k_size, rank)
                    # Local W: (k_size * i_size, rank)
                    
                    # For 2x1x2:
                    # i in {0,1}, j in {0}, k in {0,1}
                    # U_local[i*1 + 0, c]
                    # V_local[0*2 + k, c]
                    # W_local[k*2 + i, c]
                    
                    U_block = np.zeros((2*1, 4))
                    V_block = np.zeros((1*2, 4))
                    W_block = np.zeros((2*2, 4))
                    
                    # Term 0: A[0,0]B[0,0] -> C[0,0]
                    U_block[0, 0] = 1
                    V_block[0, 0] = 1
                    W_block[0, 0] = 1
                    
                    # Term 1: A[0,0]B[0,1] -> C[0,1]
                    U_block[0, 1] = 1
                    V_block[1, 1] = 1
                    W_block[1, 1] = 1
                    
                    # Term 2: A[1,0]B[0,0] -> C[1,0]
                    U_block[1, 2] = 1
                    V_block[0, 2] = 1
                    W_block[2, 2] = 1
                    
                    # Term 3: A[1,0]B[0,1] -> C[1,1]
                    U_block[1, 3] = 1
                    V_block[1, 3] = 1
                    W_block[3, 3] = 1
                    
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    # 1x2x2 Schoolbook
                    rank = 4
                    U_block = np.zeros((1*2, 4))
                    V_block = np.zeros((2*2, 4))
                    W_block = np.zeros((2*1, 4))
                    
                    # i=0, j in {0,1}, k in {0,1}
                    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
                    
                    # Term 0: A[0,0]B[0,0] -> C[0,0]
                    U_block[0, 0] = 1
                    V_block[0, 0] = 1
                    W_block[0, 0] = 1
                    
                    # Term 1: A[0,0]B[0,1] -> C[0,1]
                    U_block[0, 1] = 1
                    V_block[1, 1] = 1
                    W_block[1, 1] = 1
                    
                    # Term 2: A[0,1]B[1,0] -> C[0,0]
                    U_block[1, 2] = 1
                    V_block[2, 2] = 1
                    W_block[0, 2] = 1
                    
                    # Term 3: A[0,1]B[1,1] -> C[0,1]
                    U_block[1, 3] = 1
                    V_block[3, 3] = 1
                    W_block[1, 3] = 1
                    
                elif i_size == 1 and j_size == 1 and k_size == 2:
                    # 1x1x2 Schoolbook
                    rank = 2
                    U_block = np.zeros((1*1, 2))
                    V_block = np.zeros((1*2, 2))
                    W_block = np.zeros((2*1, 2))
                    
                    # i=0, j=0, k in {0,1}
                    # C[0,k] = A[0,0]B[0,k]
                    
                    # Term 0: A[0,0]B[0,0] -> C[0,0]
                    U_block[0, 0] = 1
                    V_block[0, 0] = 1
                    W_block[0, 0] = 1
                    
                    # Term 1: A[0,0]B[0,1] -> C[0,1]
                    U_block[0, 1] = 1
                    V_block[1, 1] = 1
                    W_block[1, 1] = 1
                    
                else:
                    # Fallback to schoolbook for unexpected sizes
                    rank = i_size * j_size * k_size
                    U_block = np.zeros((i_size * j_size, rank))
                    V_block = np.zeros((j_size * k_size, rank))
                    W_block = np.zeros((k_size * i_size, rank))
                    col = 0
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                U_block[li * j_size + lj, col] = 1
                                V_block[lj * k_size + lk, col] = 1
                                W_block[lk * i_size + li, col] = 1
                                col += 1
                
                # Place block into global matrices
                for li in range(i_size):
                    for lj in range(j_size):
                        global_i = i_start + li
                        global_j = j_start + lj
                        global_idx = global_i * m + global_j
                        for c in range(rank):
                            U[global_idx][col_offset + c] = U_block[li * j_size + lj, c]
                
                for lj in range(j_size):
                    for lk in range(k_size):
                        global_j = j_start + lj
                        global_k = k_start + lk
                        global_idx = global_j * p + global_k
                        for c in range(rank):
                            V[global_idx][col_offset + c] = V_block[lj * k_size + lk, c]
                
                for lk in range(k_size):
                    for li in range(i_size):
                        global_k = k_start + lk
                        global_i = i_start + li
                        global_idx = global_k * n + global_i
                        for c in range(rank):
                            W[global_idx][col_offset + c] = W_block[lk * i_size + li, c]
                
                col_offset += rank
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try structured Strassen
    try:
        res = _strassen_block(n, m, p)
        if _verify(res["U"], res["V"], res["W"], n, m, p):
            return res
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
