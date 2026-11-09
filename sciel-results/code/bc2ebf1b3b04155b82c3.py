import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition sizes
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U = []
    V = []
    W = []
    
    # We will build lists of (row, col, val) and then convert to dense
    # Or just build dense matrices directly since rank is small (~117)
    # Max rank 117.
    # U size 25 x 117
    # V size 25 x 117
    # W size 25 x 117
    
    # Let's determine total rank first to allocate
    # Or just append columns.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_column(u_vals, v_vals, w_vals):
        """
        u_vals: list of (row, val)
        v_vals: list of (row, val)
        w_vals: list of (row, val)
        """
        u_col = [0.0] * (n * m)
        for r, val in u_vals:
            u_col[r] += val
        v_col = [0.0] * (m * p)
        for r, val in v_vals:
            v_col[r] += val
        w_col = [0.0] * (p * n)
        for r, val in w_vals:
            w_col[r] += val
            
        cols_U.append(u_col)
        cols_V.append(v_col)
        cols_W.append(w_col)
        
    def get_global_a(i_blk, r, k_blk, c):
        return (offsets[i_blk] + r) * m + (offsets[k_blk] + c)
        
    def get_global_b(k_blk, c, j_blk, d):
        return (offsets[k_blk] + c) * p + (offsets[j_blk] + d)
        
    def get_global_w(j_blk, d, i_blk, r):
        return (offsets[j_blk] + d) * n + (offsets[i_blk] + r)
        
    # Iterate over blocks
    for i_blk in range(3):
        for k_blk in range(3):
            for j_blk in range(3):
                n_i = sizes[i_blk]
                m_k = sizes[k_blk]
                p_j = sizes[j_blk]
                
                if n_i == 2 and m_k == 2 and p_j == 2:
                    # Strassen
                    # Indices r, c, d in 0..1
                    # M1: (A11+A22)(B11+B22)
                    u = [(get_global_a(i_blk, 0, k_blk, 0), 1), (get_global_a(i_blk, 1, k_blk, 1), 1)]
                    v = [(get_global_b(k_blk, 0, j_blk, 0), 1), (get_global_b(k_blk, 1, j_blk, 1), 1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 0), 1), (get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # Wait, W index is (k, i) in local?
                    # Global W index is (d, r) -> (j_blk offset + d) * n + (i_blk offset + r)
                    # C11 is (0,0), C22 is (1,1)
                    # M1 contributes to C11 and C22?
                    # C11 = M1 + ...
                    # C22 = M1 - ...
                    # So M1 has weight 1 for C11 and 1 for C22?
                    # Yes, C11 += M1, C22 += M1.
                    # So w should have 1 at (0,0) and 1 at (1,1).
                    # get_global_w(j_blk, 0, i_blk, 0) is C(0,0)
                    # get_global_w(j_blk, 1, i_blk, 1) is C(1,1)
                    add_column(u, v, w)
                    
                    # M2: (A21+A22)B11
                    u = [(get_global_a(i_blk, 1, k_blk, 0), 1), (get_global_a(i_blk, 1, k_blk, 1), 1)]
                    v = [(get_global_b(k_blk, 0, j_blk, 0), 1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 1), 1), (get_global_w(j_blk, 1, i_blk, 1), -1)]
                    # C21 += M2, C22 -= M2
                    # C21 is (1,0) -> w(0,1)? No, C(row, col).
                    # C21 is row 1, col 0.
                    # W index: col * n + row.
                    # C21 -> col 0, row 1 -> index 1.
                    # C22 -> col 1, row 1 -> index 1*n + 1 = 6? No, local.
                    # Local W index: d * n_i + r.
                    # C21: d=0, r=1 -> 1.
                    # C22: d=1, r=1 -> 1*2 + 1 = 3.
                    # So w should be 1 at local 1, -1 at local 3.
                    # get_global_w(j_blk, 0, i_blk, 1) -> local 1.
                    # get_global_w(j_blk, 1, i_blk, 1) -> local 3.
                    add_column(u, v, w)
                    
                    # M3: A11(B12-B22)
                    u = [(get_global_a(i_blk, 0, k_blk, 0), 1)]
                    v = [(get_global_b(k_blk, 1, j_blk, 0), 1), (get_global_b(k_blk, 1, j_blk, 1), -1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 0), 1), (get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # C12 += M3, C22 += M3
                    # C12: d=1, r=0 -> local 2.
                    # C22: d=1, r=1 -> local 3.
                    # w: 1 at local 2, 1 at local 3.
                    add_column(u, v, w)
                    
                    # M4: A22(B21-B11)
                    u = [(get_global_a(i_blk, 1, k_blk, 1), 1)]
                    v = [(get_global_b(k_blk, 1, j_blk, 0), 1), (get_global_b(k_blk, 0, j_blk, 0), -1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 0), 1), (get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # C11 += M4, C21 += M4
                    # C11: local 0.
                    # C21: local 1.
                    add_column(u, v, w)
                    
                    # M5: (A11+A12)B22
                    u = [(get_global_a(i_blk, 0, k_blk, 0), 1), (get_global_a(i_blk, 0, k_blk, 1), 1)]
                    v = [(get_global_b(k_blk, 1, j_blk, 1), 1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 0), -1), (get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # C11 -= M5, C12 += M5
                    # C11: local 0.
                    # C12: local 2.
                    add_column(u, v, w)
                    
                    # M6: (A21-A11)(B11+B12)
                    u = [(get_global_a(i_blk, 1, k_blk, 0), 1), (get_global_a(i_blk, 0, k_blk, 0), -1)]
                    v = [(get_global_b(k_blk, 0, j_blk, 0), 1), (get_global_b(k_blk, 1, j_blk, 0), 1)]
                    w = [(get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # C22 += M6
                    # C22: local 3.
                    add_column(u, v, w)
                    
                    # M7: (A12-A22)(B21+B22)
                    u = [(get_global_a(i_blk, 0, k_blk, 1), 1), (get_global_a(i_blk, 1, k_blk, 1), -1)]
                    v = [(get_global_b(k_blk, 1, j_blk, 0), 1), (get_global_b(k_blk, 1, j_blk, 1), 1)]
                    w = [(get_global_w(j_blk, 0, i_blk, 0), 1), (get_global_w(j_blk, 1, i_blk, 1), 1)]
                    # C11 += M7, C22 += M7
                    # C11: local 0.
                    # C22: local 3.
                    add_column(u, v, w)
                    
                else:
                    # Schoolbook for this block
                    for r in range(n_i):
                        for c in range(m_k):
                            for d in range(p_j):
                                u = [(get_global_a(i_blk, r, k_blk, c), 1)]
                                v = [(get_global_b(k_blk, c, j_blk, d), 1)]
                                w = [(get_global_w(j_blk, d, i_blk, r), 1)]
                                add_column(u, v, w)
                                
    U = [list(col) for col in zip(*cols_U)] if cols_U else []
    V = [list(col) for col in zip(*cols_V)] if cols_V else []
    W = [list(col) for col in zip(*cols_W)] if cols_W else []
    
    # The return format expects list of lists where U[row][col]
    # My zip(*cols_U) gives rows.
    # cols_U is list of columns.
    # zip(*cols_U) gives tuples of rows.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
