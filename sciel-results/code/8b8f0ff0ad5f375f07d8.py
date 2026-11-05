import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)] # 9x34
    V = [[0] * rank for _ in range(m * p)] # 12x34
    W = [[0] * rank for _ in range(p * n)] # 12x34
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 terms
    # Indices:
    # A (3x3): rows 0-2, cols 0-2. A[i*m+j]
    # B (3x4): rows 0-2, cols 0-3. B[j*p+k]
    # C (4x3): rows 0-3, cols 0-2. C[k*n+i]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,2,3 (0*3+0, 0*3+1, 1*3+0, 1*3+1)
    # B indices: 0,1,4,5 (0*4+0, 0*4+1, 1*4+0, 1*4+1)
    # C indices: 0,1,2,3 (0*3+0, 0*3+1, 1*3+0, 1*3+1)
    
    # p1
    add_term([0,3], [1,1], [0,1], [1,1], [0,3], [1,1])
    # p2
    add_term([2,3], [1,1], [0], [1], [2,3], [1,-1])
    # p3
    add_term([0], [1], [1,5], [1,-1], [1,3], [1,1])
    # p4
    add_term([3], [1], [4,0], [1,-1], [0,2], [1,1])
    # p5
    add_term([0,1], [1,1], [5], [1], [0,1], [-1,1])
    # p6
    add_term([2,0], [1,-1], [0,1], [1,1], [3], [1])
    # p7
    add_term([1,3], [1,-1], [4,5], [1,1], [0], [1])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    # A indices: 0,1,2,3
    # B indices: 2,3,6,7 (0*4+2, 0*4+3, 1*4+2, 1*4+3)
    # C indices: 6,7,8,9 (2*3+0, 2*3+1, 3*3+0, 3*3+1)
    
    # p1
    add_term([0,3], [1,1], [2,3], [1,1], [6,9], [1,1])
    # p2
    add_term([2,3], [1,1], [2], [1], [8,9], [1,-1])
    # p3
    add_term([0], [1], [3,7], [1,-1], [7,9], [1,1])
    # p4
    add_term([3], [1], [6,2], [1,-1], [6,8], [1,1])
    # p5
    add_term([0,1], [1,1], [7], [1], [6,7], [-1,1])
    # p6
    add_term([2,0], [1,-1], [2,3], [1,1], [9], [1])
    # p7
    add_term([1,3], [1,-1], [6,7], [1,1], [6], [1])

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:4, 0:2]
    # A indices: 2,3 (0*3+2, 1*3+2) -> Wait, A[0:2, 2] is col 2 of A.
    # A[0,2]=2, A[1,2]=5.
    # B indices: 8,9,10,11 (2*4+0...2*4+3)
    # C indices: 0,1,4,5,8,9 (0*3+0, 0*3+1, 1*3+0, 1*3+1, 2*3+0, 2*3+1, 3*3+0, 3*3+1)
    # Wait, C is 4x3. C[k, i].
    # A is 2x1. B is 1x4.
    # C is 2x4.
    # C[i, k] = A[i, 0] * B[0, k].
    # Here A is A[0:2, 2] (2x1). B is B[2:3, 0:4] (1x4).
    # C is C[0:4, 0:2] (4x2).
    # C[k, i] = A[i, 2] * B[2, k].
    
    # 8 independent products
    for i in range(2):
        for k in range(4):
            a_idx = i * 3 + 2
            b_idx = 2 * 4 + k
            c_idx = k * 3 + i
            add_term([a_idx], [1], [b_idx], [1], [c_idx], [1])

    # Block 4: A[2:3, 0:3] * B[0:3, 0:4] -> C[0:4, 2:3]
    # A is 1x3. B is 3x4.
    # C is 4x1.
    # C[k, 2] = sum_j A[2, j] * B[j, k].
    # 12 independent products.
    
    for j in range(3):
        for k in range(4):
            a_idx = 2 * 3 + j
            b_idx = j * 4 + k
            c_idx = k * 3 + 2
            add_term([a_idx], [1], [b_idx], [1], [c_idx], [1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
