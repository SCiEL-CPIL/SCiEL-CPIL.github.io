import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Define the rank-44 decomposition using tensor terms.
    # Each term is a tuple of (u_dict, v_dict, w_dict)
    # Keys are local indices (i, j), (j, k), (k, i)
    # Values are coefficients
    
    strassen_terms = [
        # 1. M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        
        # 2. M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(0,1): 1, (1,1): -1}),
        
        # 3. M3 = A00 * (B01 - B11)
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}),
        
        # 4. M4 = A11 * (B10 - B00)
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}),
        
        # 5. M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (1,0): 1}),
        
        # 6. M6 = (A10 - A00) * (B00 + B01)
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        
        # 7. M7 = (A01 - A11) * (B10 + B11)
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1})
    ]
    
    schoolbook_terms_122 = [
        # 1. A00 B00 -> C00
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        
        # 2. A01 B10 -> C00
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        
        # 3. A00 B01 -> C01
        ({(0,0): 1}, {(0,1): 1}, {(1,0): 1}),
        
        # 4. A01 B11 -> C01
        ({(0,1): 1}, {(1,1): 1}, {(1,0): 1})
    ]

    # Total rank is 4 * 7 (strassen) + 4 * 4 (schoolbook) = 28 + 16 = 44
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    rank_col = 0
    
    # Helper to add a term block
    def add_block(i_off, j_off, k_off, terms, m_dim, p_dim, n_dim):
        nonlocal rank_col
        for term in terms:
            u_dict, v_dict, w_dict = term
            
            for (li, lj), coeff in u_dict.items():
                idx_u = (i_off + li) * m + (j_off + lj)
                U[idx_u, rank_col] += coeff
                
            for (lj, lk), coeff in v_dict.items():
                idx_v = (j_off + lj) * p + (k_off + lk)
                V[idx_v, rank_col] += coeff
                
            for (lk, li), coeff in w_dict.items():
                idx_w = (k_off + lk) * n + (i_off + li)
                W[idx_w, rank_col] += coeff
                
            rank_col += 1
            
            if time.time() > deadline:
                break

    # Top part: A[0:2, :] * B[:, :]
    # Split B into 4 blocks of 2x2.
    # A[0:2, 0:2] with B[0:2, 0:2] -> C[0:2, 0:2]
    add_block(0, 0, 0, strassen_terms, m, p, n)
    # A[0:2, 2:4] with B[2:4, 0:2] -> C[0:2, 0:2]
    add_block(0, 2, 0, strassen_terms, m, p, n)
    # A[0:2, 0:2] with B[0:2, 2:4] -> C[2:4, 0:2]
    add_block(0, 0, 2, strassen_terms, m, p, n)
    # A[0:2, 2:4] with B[2:4, 2:4] -> C[2:4, 0:2]
    add_block(0, 2, 2, strassen_terms, m, p, n)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Bottom part: A[2, :] * B[:, :]
    # Split B into 4 blocks of 2x2.
    # A[2, 0:2] with B[0:2, 0] -> C[0, 2]
    add_block(2, 0, 0, schoolbook_terms_122, m, p, n)
    # A[2, 2:4] with B[2:4, 0] -> C[0, 2]
    add_block(2, 2, 0, schoolbook_terms_122, m, p, n)
    # A[2, 0:2] with B[0:2, 2:4] -> C[2:4, 2]
    add_block(2, 0, 2, schoolbook_terms_122, m, p, n)
    # A[2, 2:4] with B[2:4, 2:4] -> C[2:4, 2]
    add_block(2, 2, 2, schoolbook_terms_122, m, p, n)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
