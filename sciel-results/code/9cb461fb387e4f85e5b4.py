import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    rank = 85
    U = [[0] * rank for _ in range(20)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(20)]
    
    col = 0
    
    # Part 1: [4, 4, 4] using Strassen
    # We generate the 49 terms recursively.
    
    def get_strassen_terms(k):
        """
        Returns a list of tuples (u_list, v_list, w_list)
        where each list contains (local_index, value) pairs.
        local_index is for a 2^k x 2^k matrix.
        """
        if k == 0:
            return [([(0, 1)], [(0, 1)], [(0, 1)])]
        
        sub_terms = get_strassen_terms(k-1)
        size = 2 ** (k-1)
        new_size = 2 * size
        
        terms = []
        
        # Define the 7 Strassen products
        # Each product is defined by:
        # A_blocks: list of (block_i, block_j, coeff)
        # B_blocks: list of (block_i, block_j, coeff)
        # C_blocks: list of (block_i, block_j, coeff)
        
        # M1 = (A11 + A22)(B11 + B22)
        # A: (1,1,1), (2,2,1)
        # B: (1,1,1), (2,2,1)
        # C: (1,1,1), (2,2,1)
        
        # M2 = (A21 + A22)B11
        # A: (2,1,1), (2,2,1)
        # B: (1,1,1)
        # C: (2,1,1), (2,2,-1)
        
        # M3 = A11(B12 - B22)
        # A: (1,1,1)
        # B: (1,2,1), (2,2,-1)
        # C: (1,2,1), (2,2,1)
        
        # M4 = A22(B21 - B11)
        # A: (2,2,1)
        # B: (2,1,1), (1,1,-1)
        # C: (1,1,1), (2,1,1)
        
        # M5 = (A11 + A12)B22
        # A: (1,1,1), (1,2,1)
        # B: (2,2,1)
        # C: (1,1,-1), (1,2,1)
        
        # M6 = (A21 - A11)(B11 + B12)
        # A: (2,1,1), (1,1,-1)
        # B: (1,1,1), (1,2,1)
        # C: (2,2,1)
        
        # M7 = (A12 - A22)(B21 + B22)
        # A: (1,2,1), (2,2,-1)
        # B: (2,1,1), (2,2,1)
        # C: (1,1,1)
        
        strassen_defs = [
            ([(1,1,1), (2,2,1)], [(1,1,1), (2,2,1)], [(1,1,1), (2,2,1)]),
            ([(2,1,1), (2,2,1)], [(1,1,1)], [(2,1,1), (2,2,-1)]),
            ([(1,1,1)], [(1,2,1), (2,2,-1)], [(1,2,1), (2,2,1)]),
            ([(2,2,1)], [(2,1,1), (1,1,-1)], [(1,1,1), (2,1,1)]),
            ([(1,1,1), (1,2,1)], [(2,2,1)], [(1,1,-1), (1,2,1)]),
            ([(2,1,1), (1,1,-1)], [(1,1,1), (1,2,1)], [(2,2,1)]),
            ([(1,2,1), (2,2,-1)], [(2,1,1), (2,2,1)], [(1,1,1)])
        ]
        
        for a_blocks, b_blocks, c_blocks in strassen_defs:
            for u_sub, v_sub, w_sub in sub_terms:
                u_new = []
                v_new = []
                w_new = []
                
                # Helper to add sub-term to a list of (index, value)
                def add_mapped(sub_list, blocks, target_list, is_w=False):
                    for (bi, bj, coeff) in blocks:
                        row_offset = (bi - 1) * size
                        col_offset = (bj - 1) * size
                        for (idx, val) in sub_list:
                            # idx is local index in size x size matrix
                            # Convert to (r, c)
                            r = idx // size
                            c = idx % size
                            new_r = r + row_offset
                            new_c = c + col_offset
                            new_idx = new_r * new_size + new_c
                            new_val = val * coeff
                            if new_val != 0:
                                target_list.append((new_idx, new_val))
                
                add_mapped(u_sub, a_blocks, u_new)
                add_mapped(v_sub, b_blocks, v_new)
                add_mapped(w_sub, c_blocks, w_new)
                
                terms.append((u_new, v_new, w_new))
                
        return terms
    
    terms_4x4x4 = get_strassen_terms(2)
    
    for u_list, v_list, w_list in terms_4x4x4:
        for (idx, val) in u_list:
            # Map local 4x4 index to global U index
            # Local index is i * 4 + j.
            # Global U index is i * 5 + j.
            i = idx // 4
            j = idx % 4
            U[i * 5 + j][col] = val
            
        for (idx, val) in v_list:
            # Map local 4x4 index to global V index
            # Local index is j * 4 + k.
            # Global V index is j * 5 + k.
            j = idx // 4
            k = idx % 4
            V[j * 5 + k][col] = val
            
        for (idx, val) in w_list:
            # Map local 4x4 index to global W index
            # Local index is k * 4 + i.
            # Global W index is k * 4 + i.
            # Note: W index formula is k*n + i. Here n=4.
            # So local index IS global index for the 4x4 part.
            W[idx][col] = val
            
        col += 1
        
    # Part 2: [4, 4, 1] Schoolbook
    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    # Part 3: [4, 1, 5] Schoolbook
    # A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    if time.time() > deadline:
        # Fallback to schoolbook if we ran out of time (unlikely for this size)
        # But we already filled col up to 85.
        pass
        
    return {"U": U, "V": V, "W": W}
