import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Total rank estimate: 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen Decomposition ---
    # We decompose A (4x4) and B (4x4, cols 0-3) into 2x2 blocks.
    # A blocks: A_00, A_01, A_10, A_11
    # B blocks: B_00, B_01, B_10, B_11
    # C blocks: C_00, C_01, C_10, C_11
    
    # Strassen coefficients for 4x4 level
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # Output combinations
    # C00 = P1 - P4 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P7
    
    # Define the 7 steps
    # Each step is (X_blocks, Y_blocks, W_weights)
    # X_blocks: dict of block_index -> coeff
    # Y_blocks: dict of block_index -> coeff
    # W_weights: dict of block_index -> coeff
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # P2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # P3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}), # P4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}), # P5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}), # P6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}) # P7
    ]
    
    # 2x2 Strassen coefficients
    # Term t: (U_coeffs, V_coeffs, W_coeffs)
    # U, V, W are 2x2 matrices flattened to length 4 (row-major)
    # Indices: 00, 01, 10, 11
    
    strassen_2x2 = [
        # T1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # T2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # T3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # T4
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]),
        # T5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # T6
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]),
        # T7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1])
    ]
    
    # Helper to get global indices for a block
    def get_block_indices(block_i, block_j, dim_block=2):
        # Returns list of (i, j) for the 2x2 block
        indices = []
        for di in range(dim_block):
            for dj in range(dim_block):
                indices.append((block_i * dim_block + di, block_j * dim_block + dj))
        return indices

    # Process each Strassen step
    for s_idx, (x_blocks, y_blocks, w_weights) in enumerate(steps):
        for t_idx, (u_t, v_t, w_t) in enumerate(strassen_2x2):
            # Check deadline
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # u_t, v_t, w_t are length 4.
            # Map local 2x2 indices to global indices.
            
            # For U:
            # u_t[a,b] is coeff for X_r element (a,b).
            # X_r is sum of A blocks.
            # For each local (a,b) in 0..1, 0..1:
            # The element X_r[a,b] is formed by summing A[i,j] where (i,j) is in block (I,J)
            # and the local offset matches (a,b)?
            # No. The block A[I,J] is a 2x2 matrix.
            # X_r is a 2x2 matrix.
            # X_r[a,b] = sum_{I,J} x_blocks[(I,J)] * A[I,J][a,b]
            # So for global A[i,j], the coefficient in X_r is:
            # I = i // 2, J = j // 2.
            # a = i % 2, b = j % 2.
            # coeff = x_blocks.get((I,J), 0) * u_t[a*2+b]
            
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    a = i % 2
                    b = j % 2
                    coeff = x_blocks.get((I, J), 0)
                    if coeff != 0:
                        U[i * m + j][col] += coeff * u_t[a * 2 + b]
            
            # For V:
            # v_t[b,c] is coeff for Y_r element (b,c).
            # Y_r[b,c] = sum_{J,K} y_blocks[(J,K)] * B[J,K][b,c]
            # Global B[j,k].
            # J = j // 2, K = k // 2.
            # b = j % 2, c = k % 2.
            
            for j in range(4):
                for k in range(4): # k is 0..3 for this part
                    J = j // 2
                    K = k // 2
                    b = j % 2
                    c = k % 2
                    coeff = y_blocks.get((J, K), 0)
                    if coeff != 0:
                        V[j * p + k][col] += coeff * v_t[b * 2 + c]
                        
            # For W:
            # w_t[i,k] is coeff for Z_r element (i,k).
            # Z_r contributes to C blocks.
            # C[I,K] gets weight w_weights.get((I,K), 0) * Z_r[i,k]
            # Global C[i,k].
            # I = i // 2, K = k // 2.
            # i_local = i % 2, k_local = k % 2.
            
            for i in range(4):
                for k in range(4): # k is 0..3 for this part
                    I = i // 2
                    K = k // 2
                    i_loc = i % 2
                    k_loc = k % 2
                    coeff = w_weights.get((I, K), 0)
                    if coeff != 0:
                        W[k * n + i][col] += coeff * w_t[i_loc * 2 + k_loc]
                        
            col += 1
            
    # --- Part 2: 4x4x3 Schoolbook Decomposition ---
    # B cols 4, 5, 6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
