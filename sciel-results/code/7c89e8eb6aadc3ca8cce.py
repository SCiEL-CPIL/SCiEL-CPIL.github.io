import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Pre-compute indices for efficiency and clarity
    # A is 2x4. Indices: i*4+j
    # B is 4x6. Indices: i*6+j
    # W is 12x42. W[k*2+i] corresponds to C[i,k]
    
    # Strassen formulas for 2x2 * 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M2 + M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # We have 6 such blocks.
    # Group 1: A cols 0,1. B rows 0,1.
    # Group 2: A cols 2,3. B rows 2,3.
    # Blocks in B: cols (0,1), (2,3), (4,5).
    
    col = 0
    for group in [0, 1]: # 0 for top, 1 for bottom
        for block_col in [0, 1, 2]:
            # A indices
            # local_i, local_j in 0..1
            # global_i = local_i
            # global_j = local_j + 2*group
            
            # B indices
            # local_i, local_j in 0..1
            # global_i = local_i + 2*group
            # global_j = local_j + 2*block_col
            
            # C indices
            # local_i (row 0..1), local_j (col 0..1)
            # global_i = local_i
            # global_j = local_j + 2*block_col
            # W index = global_j * 2 + global_i
            
            def A_idx(li, lj):
                return li * m + (lj + 2 * group)
            
            def B_idx(li, lj):
                return (li + 2 * group) * p + (lj + 2 * block_col)
            
            def W_idx(li, lj):
                return (lj + 2 * block_col) * n + li
            
            # M1
            U[A_idx(0,0)][col] += 1
            U[A_idx(1,1)][col] += 1
            V[B_idx(0,0)][col] += 1
            V[B_idx(1,1)][col] += 1
            W[W_idx(0,0)][col] += 1
            W[W_idx(1,1)][col] += 1
            col += 1
            
            # M2
            U[A_idx(1,0)][col] += 1
            U[A_idx(1,1)][col] += 1
            V[B_idx(0,0)][col] += 1
            W[W_idx(1,0)][col] += 1
            W[W_idx(1,1)][col] -= 1
            col += 1
            
            # M3
            U[A_idx(0,0)][col] += 1
            V[B_idx(0,1)][col] += 1
            V[B_idx(1,1)][col] -= 1
            W[W_idx(0,1)][col] += 1
            W[W_idx(1,1)][col] += 1
            col += 1
            
            # M4
            U[A_idx(1,1)][col] += 1
            V[B_idx(1,0)][col] += 1
            V[B_idx(0,0)][col] -= 1
            W[W_idx(0,0)][col] += 1
            W[W_idx(1,0)][col] += 1
            col += 1
            
            # M5
            U[A_idx(0,0)][col] += 1
            U[A_idx(0,1)][col] += 1
            V[B_idx(1,1)][col] += 1
            W[W_idx(0,0)][col] -= 1
            W[W_idx(0,1)][col] += 1
            col += 1
            
            # M6
            U[A_idx(1,0)][col] += 1
            U[A_idx(0,0)][col] -= 1
            V[B_idx(0,0)][col] += 1
            V[B_idx(0,1)][col] += 1
            W[W_idx(1,1)][col] += 1
            col += 1
            
            # M7
            U[A_idx(0,1)][col] += 1
            U[A_idx(1,1)][col] -= 1
            V[B_idx(1,0)][col] += 1
            V[B_idx(1,1)][col] += 1
            W[W_idx(0,0)][col] += 1
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
