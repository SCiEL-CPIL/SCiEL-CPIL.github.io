import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
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
        
    # Iterate over blocks
    for a in range(3):
        for b in range(3):
            for c in range(3):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block with 7 multiplications
                    # Local indices u in 0,1; v in 0,1; w in 0,1
                    
                    # M1 = (A00+A11)(B00+B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M2 = (A10+A11)B00
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = -1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M3 = A00(B01-B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M4 = A11(B10-B00)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = -1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M5 = (A00+A01)B11
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M6 = (A10-A00)(B00+B01)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+0)] = -1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # M7 = (A01-A11)(B10+B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = -1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                else:
                    # Schoolbook block
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
                                add_col(u_vec, v_vec, w_vec)
                                
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
