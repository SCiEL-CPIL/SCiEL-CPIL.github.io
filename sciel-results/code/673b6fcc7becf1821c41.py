import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 42
    R = 42
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        if col >= R:
            return
        for idx, val in zip(u_indices, u_vals):
            if val != 0:
                U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            if val != 0:
                V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen factors for 2x2x2
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (A00+A11)(B00+B11) -> C00:1, C11:1
    # U: 0:1, 3:1
    # V: 0:1, 3:1
    # W: 0:1, 3:1
    # M2: (A10+A11)B00 -> C10:1
    # U: 2:1, 3:1
    # V: 0:1
    # W: 2:1
    # M3: A00(B01-B11) -> C01:1, C11:1
    # U: 0:1
    # V: 1:1, 3:-1
    # W: 1:1, 3:1
    # M4: A11(B10-B00) -> C00:1, C10:1
    # U: 3:1
    # V: 2:1, 0:-1
    # W: 0:1, 2:1
    # M5: (A00+A01)B11 -> C00:-1, C01:1
    # U: 0:1, 1:1
    # V: 3:1
    # W: 0:-1, 1:1
    # M6: (A10-A00)(B00+B01) -> C11:1
    # U: 2:1, 0:-1
    # V: 0:1, 1:1
    # W: 3:1
    # M7: (A01-A11)(B10+B11) -> C00:1
    # U: 1:1, 3:-1
    # V: 2:1, 3:1
    # W: 0:1

    strassen_terms = [
        # (u_idx, u_val), (v_idx, v_val), (w_idx, w_val)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        for term in strassen_terms:
            u_pairs, v_pairs, w_pairs = term
            u_indices = []
            u_vals = []
            for idx, val in u_pairs:
                i, j = divmod(idx, 2)
                gi = a_row_offset + i
                gj = a_col_offset + j
                u_indices.append(gi * m + gj)
                u_vals.append(val)
                
            v_indices = []
            v_vals = []
            for idx, val in v_pairs:
                i, j = divmod(idx, 2)
                gi = b_row_offset + i
                gj = b_col_offset + j
                v_indices.append(gi * p + gj)
                v_vals.append(val)
                
            w_indices = []
            w_vals = []
            for idx, val in w_pairs:
                i, j = divmod(idx, 2)
                gi = c_row_offset + i
                gj = c_col_offset + j
                # W index is k*n + i. In Strassen, W row is k*n + i?
                # Strassen W row 0 is C00 (i=0, k=0). Row 1 is C01 (i=0, k=1).
                # Row 2 is C10 (i=1, k=0). Row 3 is C11 (i=1, k=1).
                # So i is row of C, j is col of C (which is k).
                # Global W index: k*n + i.
                # Here i is C row, j is C col (k).
                w_indices.append(j * n + gi)
                w_vals.append(val)
            
            add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals)

    # Block C11: A11 (0,0), A12 (0,2). B11 (0,0), B21 (2,0). C11 (0,0)
    # A rows 0-1, cols 0-1 and 2-3.
    # B rows 0-1 and 2-3, cols 0-1.
    # C rows 0-1, cols 0-1.
    add_strassen_block(0, 0, 0, 0, 0, 0) # A11, A12, B11, B21 -> C11
    # Wait, Strassen expects A = [[A11, A12], [A21, A22]]
    # Here A11 is at (0,0), A12 is at (0,2).
    # The Strassen block assumes A11 is top-left, A12 is top-right.
    # So a_row_offset=0, a_col_offset=0 covers A11.
    # But A12 is at col 2.
    # My add_strassen_block assumes the 2x2 block is contiguous.
    # A11 is 2x2 at (0,0). A12 is 2x2 at (0,2).
    # They are not contiguous in memory if we index by i*m+j.
    # But the Strassen logic treats them as blocks.
    # The mapping in add_strassen_block:
    # idx 0 -> (0,0) -> global (0, a_col_offset)
    # idx 1 -> (0,1) -> global (0, a_col_offset+1)
    # idx 2 -> (1,0) -> global (1, a_col_offset)
    # idx 3 -> (1,1) -> global (1, a_col_offset+1)
    # This only covers ONE 2x2 block.
    # I need to map the Strassen A blocks to the correct global positions.
    
    # Let's rewrite add_strassen_block to take explicit mappings for A, B, C blocks.
    
    def add_strassen_block_mapped(a_map, b_map, c_map):
        nonlocal col
        for term in strassen_terms:
            u_pairs, v_pairs, w_pairs = term
            u_indices = []
            u_vals = []
            for idx, val in u_pairs:
                i, j = divmod(idx, 2)
                gi, gj = a_map[i][j]
                u_indices.append(gi * m + gj)
                u_vals.append(val)
                
            v_indices = []
            v_vals = []
            for idx, val in v_pairs:
                i, j = divmod(idx, 2)
                gi, gj = b_map[i][j]
                v_indices.append(gi * p + gj)
                v_vals.append(val)
                
            w_indices = []
            w_vals = []
            for idx, val in w_pairs:
                i, j = divmod(idx, 2)
                gi, gj = c_map[i][j]
                w_indices.append(gj * n + gi)
                w_vals.append(val)
            
            add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals)

    # Define mappings for each block
    
    # C11: A11 (0,0), A12 (0,2). B11 (0,0), B21 (2,0). C11 (0,0)
    # A_map: [[(0,0), (0,2)], [(1,0), (1,2)]]
    # B_map: [[(0,0), (0,1)], [(2,0), (2,1)]]  # B11 is (0,0), B21 is (2,0)
    # C_map: [[(0,0), (0,1)], [(1,0), (1,1)]]
    
    a_map_c11 = [[(0,0), (0,2)], [(1,0), (1,2)]]
    b_map_c11 = [[(0,0), (0,1)], [(2,0), (2,1)]]
    c_map_c11 = [[(0,0), (0,1)], [(1,0), (1,1)]]
    add_strassen_block_mapped(a_map_c11, b_map_c11, c_map_c11)
    
    # C12: A11 (0,0), A12 (0,2). B12 (0,2), B22 (2,2). C12 (0,2)
    a_map_c12 = [[(0,0), (0,2)], [(1,0), (1,2)]]
    b_map_c12 = [[(0,2), (0,3)], [(2,2), (2,3)]]
    c_map_c12 = [[(0,2), (0,3)], [(1,2), (1,3)]]
    add_strassen_block_mapped(a_map_c12, b_map_c12, c_map_c12)
    
    # C13: Rank 8
    # A11 (0,0) x B13 (0,4) -> C13 (0,4)
    # A12 (0,2) x B23 (2,4) -> C13 (0,4)
    # A11 is 2x2. B13 is 2x1.
    # Terms:
    # A11[0,0]*B13[0,0] -> C13[0,0]
    # A11[0,1]*B13[1,0] -> C13[0,0]
    # A11[1,0]*B13[0,0] -> C13[1,0]
    # A11[1,1]*B13[1,0] -> C13[1,0]
    # A12[0,0]*B23[0,0] -> C13[0,0]
    # A12[0,1]*B23[1,0] -> C13[0,0]
    # A12[1,0]*B23[0,0] -> C13[1,0]
    # A12[1,1]*B23[1,0] -> C13[1,0]
    
    # C21: Rank 8
    # A21 (2,0) x B11 (0,0) -> C21 (2,0)
    # A22 (2,2) x B21 (2,0) -> C21 (2,0)
    
    # C22: Rank 8
    # A21 (2,0) x B12 (0,2) -> C22 (2,2)
    # A22 (2,2) x B22 (2,2) -> C22 (2,2)
    
    # C23: Rank 4
    # A21 (2,0) x B13 (0,4) -> C23 (2,4)
    # A22 (2,2) x B23 (2,4) -> C23 (2,4)
    
    # We'll implement these schoolbook blocks directly.
    
    def add_schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols are lists of indices
        # b_rows, b_cols are lists of indices
        # c_rows, c_cols are lists of indices
        # This assumes 1D blocks (vectors) or simple 2x1, 1x2, 1x1.
        # For 2x2 x 2x1, we iterate.
        pass

    # Let's just use the generic add_term for the schoolbook parts.
    
    # C13 (2x1)
    # A11 (2x2) * B13 (2x1)
    for i in range(2):
        for j in range(2):
            add_term([(i*4 + j)], [(j*5 + 4)], [(4*3 + i)], [1], [1], [1])
            # A11[i,j] is at i, j. B13[j,0] is at j, 4. C13[i,0] is at 4, i.
            # W index: k*n + i = 4*3 + i = 12+i.
            
    # A12 (2x2) * B23 (2x1)
    for i in range(2):
        for j in range(2):
            add_term([(i*4 + (2+j))], [((2+j)*5 + 4)], [(4*3 + i)], [1], [1], [1])
            # A12[i,j] is at i, 2+j. B23[j,0] is at 2+j, 4.
            
    # C21 (1x2)
    # A21 (1x2) * B11 (2x2)
    for j in range(2):
        for k in range(2):
            add_term([(2*4 + j)], [(j*5 + k)], [(k*3 + 2)], [1], [1], [1])
            # A21[0,j] is at 2, j. B11[j,k] is at j, k. C21[0,k] is at k, 2.
            
    # A22 (1x2) * B21 (2x2)
    for j in range(2):
        for k in range(2):
            add_term([(2*4 + (2+j))], [((2+j)*5 + k)], [(k*3 + 2)], [1], [1], [1])
            
    # C22 (1x2)
    # A21 (1x2) * B12 (2x2)
    for j in range(2):
        for k in range(2):
            add_term([(2*4 + j)], [(j*5 + (2+k))], [((2+k)*3 + 2)], [1], [1], [1])
            
    # A22 (1x2) * B22 (2x2)
    for j in range(2):
        for k in range(2):
            add_term([(2*4 + (2+j))], [((2+j)*5 + (2+k))], [((2+k)*3 + 2)], [1], [1], [1])
            
    # C23 (1x1)
    # A21 (1x2) * B13 (2x1)
    for j in range(2):
        add_term([(2*4 + j)], [(j*5 + 4)], [(4*3 + 2)], [1], [1], [1])
        
    # A22 (1x2) * B23 (2x1)
    for j in range(2):
        add_term([(2*4 + (2+j))], [((2+j)*5 + 4)], [(4*3 + 2)], [1], [1], [1])
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
