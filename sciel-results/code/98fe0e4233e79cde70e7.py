import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Base Strassen rank-7 decomposition for 2x2x2
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # We define the 7 products explicitly
    # A_coeffs: 4x7 matrix (row: A_index, col: term)
    # B_coeffs: 4x7 matrix (row: B_index, col: term)
    # C_coeffs: 4x7 matrix (row: C_index, col: term)
    
    A_base = np.array([
        [1, 0, 1, 0, 1, -1, 1], # A00
        [0, 1, 0, 1, 1, 1, -1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 0]   # A11
    ])
    
    B_base = np.array([
        [1, 1, 1, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 1, 1, 1],  # B01
        [1, 0, 0, 1, 0, 0, 1],  # B10
        [1, -1, -1, 0, 1, 0, 0] # B11
    ])
    
    C_base = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # We have p=8 columns in B, grouped into p//2 = 4 blocks of size 2.
    num_blocks = p // 2
    
    total_rank = 7 * num_blocks
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Populate the matrices
    # U is same for all blocks (since A is same for all blocks in this decomposition)
    # But in the tensor sum, U is multiplied by V_t and W_t.
    # The term t corresponds to block b and strassen product r.
    # U[:, t] = A_base[:, r]
    # V[:, t] has non-zeros only in the rows corresponding to block b.
    # W[:, t] has non-zeros only in the rows corresponding to block b.
    
    for b in range(num_blocks):
        offset = b * 7
        # U part: A is shared, so A_base[:, r] goes into U[:, offset+r]
        U[:, offset:offset+7] = A_base
        
        # V part: B is block-diagonal
        # V has shape (m*p, rank) = (2*8, 28) = (16, 28)
        # For block b, the B indices are 2*b and 2*b+1.
        # B_base has shape (4, 7).
        # We need to map B_base indices (0..3) to global B indices.
        # B_global index = j * p + k.
        # Here j is 0..1 (row of B), k is 0..7 (col of B).
        # B_base row index = j * 2 + k_sub.
        # So for B_base row rb (0..3):
        # j = rb // 2, k_sub = rb % 2.
        # Global k = 2*b + k_sub.
        # Global V index = j * p + global_k.
        
        for rb in range(4):
            j = rb // 2
            k_sub = rb % 2
            global_k = 2 * b + k_sub
            v_idx = j * p + global_k
            V[v_idx, offset:offset+7] = B_base[rb, :]
            
        # W part: C is block-diagonal
        # W has shape (p*n, rank) = (8*2, 28) = (16, 28)
        # Convention: W[k*n + i, r]
        # C_base row index = i * 2 + k_sub? 
        # Wait, C_base was defined with rows 0..3 corresponding to C00, C01, C10, C11.
        # In standard C matrix (2x2):
        # C00 is (0,0), C01 is (0,1), C10 is (1,0), C11 is (1,1).
        # So row rc in C_base: i = rc // 2, k_sub = rc % 2.
        # Global k = 2*b + k_sub.
        # Global W index = global_k * n + i.
        
        for rc in range(4):
            i = rc // 2
            k_sub = rc % 2
            global_k = 2 * b + k_sub
            w_idx = global_k * n + i
            W[w_idx, offset:offset+7] = C_base[rc, :]
            
    # Check time
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank by checking linear dependencies in columns of U, V, W simultaneously?
    # Since U is block-repeated, its columns within a block are independent, and blocks are identical in U.
    # This implies U has rank 7.
    # V and W have block structure.
    # A global rank reduction would require combining columns from different blocks.
    # This is generally not possible for exact rank in this direct sum structure unless the tensor
    # has a lower border rank or a specific shared structure not captured by direct sum.
    # Given the constraints and known results, 28 is likely minimal for this specific decomposition family.
    # We return the 28-rank solution.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
