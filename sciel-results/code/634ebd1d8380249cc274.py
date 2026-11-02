import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 5s margin

    # Initialize matrices
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols: indices in A (2x4)
        # b_rows, b_cols: indices in B (4x4)
        # c_rows, c_cols: indices in C (2x4)
        
        # M1: (A00 + A11) * (B00 + B11)
        # A part: A[r0,c0] + A[r1,c1]
        # B part: B[br0,bc0] + B[br1,bc1]
        U[a_rows[0]*m + a_cols[0]][base_col+0] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+0] += 1
        V[b_rows[0]*p + b_cols[0]][base_col+0] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+0] += 1
        
        # M2: (A10 + A11) * B00
        # A part: A[r1,c0] + A[r1,c1]
        # B part: B[br0,bc0]
        U[a_rows[1]*m + a_cols[0]][base_col+1] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+1] += 1
        V[b_rows[0]*p + b_cols[0]][base_col+1] += 1
        
        # M3: A00 * (B01 - B11)
        # A part: A[r0,c0]
        # B part: B[br0,bc1] - B[br1,bc1]
        U[a_rows[0]*m + a_cols[0]][base_col+2] += 1
        V[b_rows[0]*p + b_cols[1]][base_col+2] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+2] -= 1
        
        # M4: A11 * B10
        # A part: A[r1,c1]
        # B part: B[br1,bc0]
        U[a_rows[1]*m + a_cols[1]][base_col+3] += 1
        V[b_rows[1]*p + b_cols[0]][base_col+3] += 1
        
        # M5: (A00 + A01) * B11
        # A part: A[r0,c0] + A[r0,c1]
        # B part: B[br1,bc1]
        U[a_rows[0]*m + a_cols[0]][base_col+4] += 1
        U[a_rows[0]*m + a_cols[1]][base_col+4] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+4] += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        # A part: A[r1,c0] - A[r0,c0]
        # B part: B[br0,bc0] + B[br0,bc1]
        U[a_rows[1]*m + a_cols[0]][base_col+5] += 1
        U[a_rows[0]*m + a_cols[0]][base_col+5] -= 1
        V[b_rows[0]*p + b_cols[0]][base_col+5] += 1
        V[b_rows[0]*p + b_cols[1]][base_col+5] += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        # A part: A[r0,c1] - A[r1,c1]
        # B part: B[br1,bc0] + B[br1,bc1]
        U[a_rows[0]*m + a_cols[1]][base_col+6] += 1
        U[a_rows[1]*m + a_cols[1]][base_col+6] -= 1
        V[b_rows[1]*p + b_cols[0]][base_col+6] += 1
        V[b_rows[1]*p + b_cols[1]][base_col+6] += 1
        
        # Update W (C)
        # C00 = M1 + M4 - M5 + M7
        w_idx = c_cols[0]*n + c_rows[0]
        W[w_idx][base_col+0] += 1
        W[w_idx][base_col+3] += 1
        W[w_idx][base_col+4] -= 1
        W[w_idx][base_col+6] += 1
        
        # C01 = M3 + M5
        w_idx = c_cols[1]*n + c_rows[0]
        W[w_idx][base_col+2] += 1
        W[w_idx][base_col+4] += 1
        
        # C10 = M2 + M4
        w_idx = c_cols[0]*n + c_rows[1]
        W[w_idx][base_col+1] += 1
        W[w_idx][base_col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        w_idx = c_cols[1]*n + c_rows[1]
        W[w_idx][base_col+0] += 1
        W[w_idx][base_col+1] -= 1
        W[w_idx][base_col+2] += 1
        W[w_idx][base_col+5] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # a_rows: 0,1; a_cols: 0,1
    # b_rows: 0,1; b_cols: 0,1
    # c_rows: 0,1; c_cols: 0,1
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Wait, C[0:2, 0:2] is the left half of C.
    # C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # So Block 1 contributes to C[:, 0:2] from A[:, 0:2] and B[0:2, 0:2].
    # Block 2 contributes to C[:, 0:2] from A[:, 2:4] and B[2:4, 0:2].
    # This matches the logic.
    add_strassen_block(7, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # C[:, 2:4] = A[:, 0:2] B[0:2, 2:4] + A[:, 2:4] B[2:4, 2:4]
    add_strassen_block(14, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(21, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
