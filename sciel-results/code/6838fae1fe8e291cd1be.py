import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    # Rank 28 Strassen-based solution for 2x4x4 matrix multiplication
    # Split the 4x4 B matrix into four 2x2 blocks along rows, and C into four 2x2 blocks along columns
    # Each 2x2x2 sub-multiplication uses Strassen (rank 7), total rank = 4*7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 28)

    # Strassen's 7 multiplications for 2x2 matrix mult:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For our 2x4x4 problem:
    # A is 2x4, B is 4x4, C is 2x4
    # We split B into B_left (cols 0,1) and B_right (cols 2,3), each 4x2
    # We split C into C_left (cols 0,1) and C_right (cols 2,3), each 2x2
    # Actually, let's think of it differently:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split j into {0,1} and {2,3}, split k into {0,1} and {2,3}
    # C[i,0:2] = A[i,0:2] @ B[0:2,0:2] + A[i,2:4] @ B[2:4,0:2]
    # C[i,2:4] = A[i,0:2] @ B[0:2,2:4] + A[i,2:4] @ B[2:4,2:4]
    # So we have 4 block multiplications of 2x2 @ 2x2, each done with Strassen (rank 7)

    strassen_steps = [
        # M1 = (A11+A22)(B11+B22)
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2 = (A21+A22)B11
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        # M3 = A11(B21-B22)
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        # M4 = A22(B12-B21)
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        # M5 = (A11+A12)B22
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        # M6 = (A21-A11)(B11+B12)
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        # M7 = (A12-A22)(B21+B22)
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
    ]

    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        """Add one 2x2x2 Strassen block.
        a_col_start: starting column in A (0 or 2)
        b_row_start: starting row in B (0 or 2)
        b_col_start: starting column in B (0 or 2)
        c_col_start: starting column in C (0 or 2)
        col_offset: starting column in the rank dimension
        """
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V[global_idx][col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    # Four 2x2x2 blocks:
    # Block 1: A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_block(0, 0, 0, 0, 0)
    # Block 2: A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    add_block(0, 0, 2, 2, 7)
    # Block 3: A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    add_block(2, 2, 0, 0, 14)
    # Block 4: A[0:2,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
    add_block(2, 2, 2, 2, 21)

    # Verify rank 28 solution with exact integer arithmetic
    def verify(U, V, W, n, m, p, rank):
        # For matrix multiplication tensor M_{n,m,p}:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W, n, m, p, rank):
        # Fallback: naive rank 32
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    r = i * p * m + k * m + j
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        if not verify(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
