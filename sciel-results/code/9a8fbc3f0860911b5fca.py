import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 4x4x4 core: 7 terms (Strassen)
    # 4x4x1: 16 terms (A[0:4,0:4] * B[0:4,4])
    # 4x1x4: 16 terms (A[0:4,4] * B[4,0:4])
    # 4x1x1: 4 terms (A[0:4,4] * B[4,4])
    # Total: 7 + 16 + 16 + 4 = 43
    rank = 43
    
    U = [[0] * rank for _ in range(n * m)]  # (20, rank)
    V = [[0] * rank for _ in range(m * p)]  # (25, rank)
    W = [[0] * rank for _ in range(p * n)]  # (20, rank)
    
    # Strassen's 2x2x2 factors (rank 7)
    # Each factor is (4, 7) where rows are 0,1,2,3 (flattened 2x2)
    # Term 0: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    # Term 1: (a10+a11)b00 -> c10 += 1, c11 -= 1
    # Term 2: a00(b01-b11) -> c01 += 1, c11 += 1
    # Term 3: a11(b10-b00) -> c00 += 1, c10 += 1
    # Term 4: (a00+a01)b11 -> c00 -= 1, c01 += 1
    # Term 5: (a10-a00)(b00+b01) -> c11 += 1
    # Term 6: (a01-a11)(b10+b11) -> c00 += 1
    
    s_u = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1],   # a11
    ]
    s_v = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 1, 0],    # b01
        [0, 0, 0, 1, 0, 0, 1],    # b10
        [1, 0, -1, 0, 1, 0, 1],   # b11
    ]
    s_w = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0],   # c11
    ]
    
    # 4x4x4 core: Use Strassen on 2x2 blocks of the 4x4 matrices
    # We have 4x4 = (2x2) x (2x2) x (2x2) blocks
    # Block (br, bc) in A, (br, bc) in B, (br, bc) in C
    # For each pair of 2x2 blocks, we need 7 multiplications
    
    # A is 4x5, B is 5x5, C is 5x4
    # The 4x4 core uses A[0:4,0:4], B[0:4,0:4], C[0:4,0:4]
    
    # Process each 2x2 block pair
    # Block indices: (0,0), (0,1), (1,0), (1,1) for 2x2 blocks within 4x4
    block_pairs = [
        ((0, 0), (0, 0)),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        ((0, 0), (0, 1)),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        ((0, 0), (1, 0)),  # A[0:2,0:2] * B[2:4,0:2] -> C[2:4,0:2]
        ((0, 0), (1, 1)),  # A[0:2,0:2] * B[2:4,2:4] -> C[2:4,2:4]
        ((0, 1), (0, 0)),  # A[0:2,2:4] * B[0:2,0:2] -> C[0:2,0:2]
        ((0, 1), (0, 1)),  # A[0:2,2:4] * B[0:2,2:4] -> C[0:2,2:4]
        ((0, 1), (1, 0)),  # A[0:2,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        ((0, 1), (1, 1)),  # A[0:2,2:4] * B[2:4,2:4] -> C[2:4,2:4]
        ((1, 0), (0, 0)),  # A[2:4,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        ((1, 0), (0, 1)),  # A[2:4,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        ((1, 0), (1, 0)),  # A[2:4,0:2] * B[2:4,0:2] -> C[2:4,0:2]
        ((1, 0), (1, 1)),  # A[2:4,0:2] * B[2:4,2:4] -> C[2:4,2:4]
        ((1, 1), (0, 0)),  # A[2:4,2:4] * B[0:2,0:2] -> C[0:2,0:2]
        ((1, 1), (0, 1)),  # A[2:4,2:4] * B[0:2,2:4] -> C[0:2,2:4]
        ((1, 1), (1, 0)),  # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        ((1, 1), (1, 1)),  # A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    ]
    
    col = 0
    for (a_br, a_bc), (b_br, b_bc) in block_pairs:
        c_br = a_br  # C row block
        c_bc = b_bc  # C col block
        
        for k in range(7):
            # U: A block (a_br, a_bc), local position k
            for lr in range(2):
                for lc in range(2):
                    u_val = s_u[lr * 2 + lc][k]
                    if u_val != 0:
                        global_i = 2 * a_br + lr
                        global_j = 2 * a_bc + lc
                        U[global_i * 5 + global_j][col] += u_val
            
            # V: B block (b_br, b_bc), local position k
            for lr in range(2):
                for lc in range(2):
                    v_val = s_v[lr * 2 + lc][k]
                    if v_val != 0:
                        global_j = 2 * b_br + lr
                        global_k = 2 * b_bc + lc
                        V[global_j * 5 + global_k][col] += v_val
            
            # W: C block (c_br, c_bc), local position k
            for lr in range(2):
                for lc in range(2):
                    w_val = s_w[lr * 2 + lc][k]
                    if w_val != 0:
                        global_k = 2 * c_bc + lc
                        global_i = 2 * c_br + lr
                        W[global_k * 4 + global_i][col] += w_val
            
            col += 1
    
    # 4x4x1 part: A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # 4x1x4 part: A[0:4,4] * B[4,0:4] -> C[0:4,0:4]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    # 4x1x1 part: A[0:4,4] * B[4,4] -> C[0:4,4]
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
