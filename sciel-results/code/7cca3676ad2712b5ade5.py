import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    # Pre-calculate block ranks
    # Block (a,b,c) has size parts[a] x parts[b] x parts[c]
    # If 2x2x2, rank 7. Else rank = prod(sizes).
    
    # We need to map global indices to local indices within blocks.
    # Global i in 0..4. 
    # Block 0 covers i=0,1. Block 1 covers i=2,3. Block 2 covers i=4.
    
    # We will build U, V, W lists of lists.
    # Shapes: (25, total_rank), (25, total_rank), (25, total_rank)
    # Since we don't know total_rank until we iterate, we can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a column
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
                
                # Determine offsets
                # Offset for index 0 is 0
                # Offset for index 1 is 2
                # Offset for index 2 is 4
                offsets = [0, 2, 4]
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block
                    # Local indices u in 0,1; v in 0,1; w in 0,1
                    # Global i = off_a + u, j = off_b + v, k = off_c + w
                    
                    # Strassen Factors
                    # Factor 0: M1
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A00, A11
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    # V: B00, B11
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    # W: C00, C11
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 1: M2
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A10, A11
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    # V: B00
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    # W: C10 (+1), C11 (-1)
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = -1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 2: M3
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A00
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    # V: B01 (+1), B11 (-1)
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = -1
                    # W: C01 (+1), C11 (+1)
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 3: M4
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A11
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    # V: B10 (+1), B00 (-1)
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = -1
                    # W: C00 (+1), C10 (+1)
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 4: M5
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A00 (+1), A01 (+1)
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    # V: B11
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    # W: C00 (-1), C01 (+1)
                    w_vec[(off_c+0)*5 + (off_a+0)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 5: M6
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A10 (+1), A00 (-1)
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+0)] = -1
                    # V: B00 (+1), B01 (+1)
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    # W: C11 (+1)
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 6: M7
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    # U: A01 (+1), A11 (-1)
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = -1
                    # V: B10 (+1), B11 (+1)
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    # W: C00 (+1)
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
