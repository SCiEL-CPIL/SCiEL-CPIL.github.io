import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_decomp(n):
    """
    Recursively generates U, V, W matrices for n x n x n matrix multiplication
    using Strassen's algorithm.
    n must be a power of 2.
    Returns U, V, W as numpy arrays of shape (n*n, rank).
    """
    if n == 1:
        return np.array([[1]]), np.array([[1]]), np.array([[1]])
    
    m = n // 2
    U_m, V_m, W_m = _strassen_decomp(m)
    rank_m = U_m.shape[1]
    rank_n = 7 * rank_m
    
    U_n = np.zeros((n*n, rank_n))
    V_n = np.zeros((n*n, rank_n))
    W_n = np.zeros((n*n, rank_n))
    
    # Define the 7 Strassen terms
    # Each term is a list of (coeff_a, block_a, coeff_b, block_b)
    # block is (row_block, col_block)
    # We need to map these to the U and V contributions.
    
    # Strassen Formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 + B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output Formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # For each term t in 0..6, we define which A blocks and B blocks are involved.
    # A_blocks: list of (coeff, br, bc)
    # B_blocks: list of (coeff, br, bc)
    
    terms = [
        # M1
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        # M2
        ([(1, 1, 0), (1, 1, 1)], [(1, 0, 0)]),
        # M3
        ([(1, 0, 0)], [(1, 1, 0), (-1, 1, 1)]),
        # M4
        ([(1, 1, 1)], [(1, 0, 1), (1, 1, 1)]),
        # M5
        ([(1, 0, 0), (1, 0, 1)], [(1, 1, 1)]),
        # M6
        ([(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)]),
        # M7
        ([(1, 0, 1), (-1, 1, 1)], [(1, 1, 0), (1, 1, 1)])
    ]
    
    # Output mapping: for each output block (br, bc), which M terms contribute and with what coeff
    # C00 (0,0): M1(1), M4(-1), M5(1), M7(1)
    # C01 (0,1): M3(1), M4(1)
    # C10 (1,0): M2(1), M5(1)
    # C11 (1,1): M1(1), M2(-1), M3(1), M6(1)
    
    out_map = {
        (0, 0): [(0, 1), (3, -1), (4, 1), (6, 1)],
        (0, 1): [(2, 1), (3, 1)],
        (1, 0): [(1, 1), (4, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Fill U_n
    for t in range(7):
        a_blocks, b_blocks = terms[t]
        for coeff_a, br_a, bc_a in a_blocks:
            # Iterate over all rows in this A block
            for lr in range(m):
                for lc in range(m):
                    global_i = (br_a * m + lr) * n + (bc_a * m + lc)
                    local_idx = lr * m + lc
                    # Add U_m[local_idx] * coeff_a to U_n[global_i, t*rank_m : (t+1)*rank_m]
                    U_n[global_i, t*rank_m:(t+1)*rank_m] += coeff_a * U_m[local_idx]
                    
    # Fill V_n
    for t in range(7):
        a_blocks, b_blocks = terms[t]
        for coeff_b, br_b, bc_b in b_blocks:
            for lr in range(m):
                for lc in range(m):
                    global_j = (br_b * m + lr) * n + (bc_b * m + lc)
                    local_idx = lr * m + lc
                    V_n[global_j, t*rank_m:(t+1)*rank_m] += coeff_b * V_m[local_idx]
                    
    # Fill W_n
    # W index is k*n + i.
    # Local W index is k_local*m + i_local.
    # For each output block (br_c, bc_c), we have contributions from certain M terms.
    # br_c is row block of C (i block), bc_c is col block of C (k block).
    # Wait, C is n x n. i is row, k is col.
    # C[i, k].
    # Block (br_c, bc_c) means i in br_c, k in bc_c.
    # Global i = br_c*m + i_local.
    # Global k = bc_c*m + k_local.
    # Global W row index = k*n + i.
    
    for br_c in range(2):
        for bc_c in range(2):
            # Get contributions for this output block
            contribs = out_map[(br_c, bc_c)]
            for lr in range(m): # i_local
                for lc in range(m): # k_local
                    global_i = br_c * m + lr
                    global_k = bc_c * m + lc
                    global_w_idx = global_k * n + global_i
                    local_w_idx = lc * m + lr # Note: W_m index is k_local*m + i_local. Here lc is k_local, lr is i_local.
                    
                    for t_idx, coeff_w in contribs:
                        # t_idx is the M term index (0..6)
                        # coeff_w is the coefficient
                        W_n[global_w_idx, t_idx*rank_m:(t_idx+1)*rank_m] += coeff_w * W_m[local_w_idx]
                        
    return U_n, V_n, W_n

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # We use two 4x4x4 Strassen decompositions
    U4, V4, W4 = _strassen_decomp(4)
    rank4 = U4.shape[1] # 49
    
    total_rank = 2 * rank4 # 98
    
    U_final = np.zeros((n*m, total_rank))
    V_final = np.zeros((m*p, total_rank))
    W_final = np.zeros((p*n, total_rank))
    
    # First half: B cols 0-3, C cols 0-3
    # U is same
    U_final[:, 0:rank4] = U4
    
    # V: B is 4x4 for this part.
    # V4 rows are indexed by j*4 + k (j=0..3, k=0..3)
    # In V_final, rows are indexed by j*8 + k_global.
    # For first half, k_global = k.
    # So V_final[j*8 + k, 0:rank4] = V4[j*4 + k, :]
    for j in range(m):
        for k in range(4):
            V_final[j*8 + k, 0:rank4] = V4[j*4 + k]
            
    # W: C is 4x4 for this part.
    # W4 rows are indexed by k*4 + i (k=0..3, i=0..3)
    # In W_final, rows are indexed by k_global*4 + i.
    # For first half, k_global = k.
    # So W_final[k*4 + i, 0:rank4] = W4[k*4 + i, :]
    # Note: W_final row index is k*n + i. n=4.
    for k in range(4):
        for i in range(n):
            W_final[k*n + i, 0:rank4] = W4[k*4 + i]
            
    # Second half: B cols 4-7, C cols 4-7
    # U is same
    U_final[:, rank4:2*rank4] = U4
    
    # V: B is 4x4 for this part, but cols are 4-7.
    # V4 rows are indexed by j*4 + k_local (k_local=0..3)
    # In V_final, rows are indexed by j*8 + k_global.
    # k_global = k_local + 4.
    for j in range(m):
        for k_local in range(4):
            k_global = k_local + 4
            V_final[j*8 + k_global, rank4:2*rank4] = V4[j*4 + k_local]
            
    # W: C is 4x4 for this part, but cols are 4-7.
    # W4 rows are indexed by k_local*4 + i (k_local=0..3)
    # In W_final, rows are indexed by k_global*4 + i.
    # k_global = k_local + 4.
    for k_local in range(4):
        k_global = k_local + 4
        for i in range(n):
            W_final[k_global*n + i, rank4:2*rank4] = W4[k_local*4 + i]
            
    # Convert to lists of lists for the required format
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use two 4x4x4 Strassen decompositions
    U4, V4, W4 = _strassen_decomp(4)
    rank4 = U4.shape[1] # 49
    
    total_rank = 2 * rank4 # 98
    
    U_final = np.zeros((n*m, total_rank))
    V_final = np.zeros((m*p, total_rank))
    W_final = np.zeros((p*n, total_rank))
    
    # First half: B cols 0-3, C cols 0-3
    U_final[:, 0:rank4] = U4
    
    for j in range(m):
        for k in range(4):
            V_final[j*8 + k, 0:rank4] = V4[j*4 + k]
            
    for k in range(4):
        for i in range(n):
            W_final[k*n + i, 0:rank4] = W4[k*4 + i]
            
    # Second half: B cols 4-7, C cols 4-7
    U_final[:, rank4:2*rank4] = U4
    
    for j in range(m):
        for k_local in range(4):
            k_global = k_local + 4
            V_final[j*8 + k_global, rank4:2*rank4] = V4[j*4 + k_local]
            
    for k_local in range(4):
        k_global = k_local + 4
        for i in range(n):
            W_final[k_global*n + i, rank4:2*rank4] = W4[k_local*4 + i]
            
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
