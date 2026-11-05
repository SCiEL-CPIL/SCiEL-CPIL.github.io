import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def check_time():
        return time.time() > deadline
    
    def strassen_2x2x2(base_a, base_b, base_c):
        """Add Strassen rank-7 decomposition for 2x2x2 block at given offsets"""
        # M1 = (A00+A11)(B00+B11)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+0)*5 + (base_b+0)] = 1
        u_vec[(base_a+1)*5 + (base_b+1)] = 1
        v_vec[(base_b+0)*5 + (base_c+0)] = 1
        v_vec[(base_b+1)*5 + (base_c+1)] = 1
        w_vec[(base_c+0)*5 + (base_a+0)] = 1
        w_vec[(base_c+1)*5 + (base_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M2 = (A10+A11)B00
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+1)*5 + (base_b+0)] = 1
        u_vec[(base_a+1)*5 + (base_b+1)] = 1
        v_vec[(base_b+0)*5 + (base_c+0)] = 1
        w_vec[(base_c+0)*5 + (base_a+1)] = 1
        w_vec[(base_c+1)*5 + (base_a+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # M3 = A00(B01-B11)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+0)*5 + (base_b+0)] = 1
        v_vec[(base_b+0)*5 + (base_c+1)] = 1
        v_vec[(base_b+1)*5 + (base_c+1)] = -1
        w_vec[(base_c+1)*5 + (base_a+0)] = 1
        w_vec[(base_c+1)*5 + (base_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M4 = A11(B10-B00)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+1)*5 + (base_b+1)] = 1
        v_vec[(base_b+1)*5 + (base_c+0)] = 1
        v_vec[(base_b+0)*5 + (base_c+0)] = -1
        w_vec[(base_c+0)*5 + (base_a+0)] = 1
        w_vec[(base_c+0)*5 + (base_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M5 = (A00+A01)B11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+0)*5 + (base_b+0)] = 1
        u_vec[(base_a+0)*5 + (base_b+1)] = 1
        v_vec[(base_b+1)*5 + (base_c+1)] = 1
        w_vec[(base_c+0)*5 + (base_a+0)] = -1
        w_vec[(base_c+1)*5 + (base_a+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M6 = (A10-A00)(B00+B01)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+1)*5 + (base_b+0)] = 1
        u_vec[(base_a+0)*5 + (base_b+0)] = -1
        v_vec[(base_b+0)*5 + (base_c+0)] = 1
        v_vec[(base_b+0)*5 + (base_c+1)] = 1
        w_vec[(base_c+1)*5 + (base_a+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # M7 = (A01-A11)(B10+B11)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(base_a+0)*5 + (base_b+1)] = 1
        u_vec[(base_a+1)*5 + (base_b+1)] = -1
        v_vec[(base_b+1)*5 + (base_c+0)] = 1
        v_vec[(base_b+1)*5 + (base_c+1)] = 1
        w_vec[(base_c+0)*5 + (base_a+0)] = 1
        add_col(u_vec, v_vec, w_vec)
    
    def schoolbook_block(base_a, base_b, base_c, size_a, size_b, size_c):
        """Add schoolbook decomposition for a block"""
        for u in range(size_a):
            for v in range(size_b):
                for w in range(size_c):
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    gi = base_a + u
                    gj = base_b + v
                    gk = base_c + w
                    
                    u_vec[gi*5 + gj] = 1
                    v_vec[gj*5 + gk] = 1
                    w_vec[gk*5 + gi] = 1
                    add_col(u_vec, v_vec, w_vec)
    
    # Iterate over blocks
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if check_time():
                    break
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    strassen_2x2x2(off_a, off_b, off_c)
                else:
                    schoolbook_block(off_a, off_b, off_c, size_a, size_b, size_c)
    
    # Convert lists of columns to lists of rows
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
