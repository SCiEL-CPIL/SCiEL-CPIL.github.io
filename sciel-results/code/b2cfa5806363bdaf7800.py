import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 2, 2, 15
    
    # We will use 7 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Schoolbook, rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Correct Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i,j], V[j,k], W[k,i]
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output reconstruction:
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    U_strassen = [
        [1, 0, 1, 0, 0, -1, 0], # A00: M1, M3, M6
        [0, 0, 0, 0, 1, 0, 1],  # A01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],  # A10: M2, M6
        [1, 1, 0, 1, 0, 0, -1]  # A11: M1, M2, M4, M7
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00: M1, M2, M4, M6
        [0, 0, 1, 0, 0, 1, 0],  # B01: M3, M6
        [0, 0, 0, 1, 0, 0, 1],  # B10: M4, M7
        [1, 0, -1, 0, 1, 0, 1]  # B11: M1, M3, M5, M7
    ]
    
    W_strassen = [
        [1, 0, 0, -1, 0, -1, 1], # C00: M1, M4, M6, M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2, M4
        [1, -1, 1, 0, 1, 0, 0]   # C11: M1, M2, M3, M5
    ]

    # Schoolbook for 2x2x1
    # Rank 4
    # U: 4x4 identity
    # V: 4x4 identity (mapping A_ij to B_jk? No, V is (j,k). Here k is fixed to 14.
    #    We need V[j, 14] to pick up A_ij? No, U picks up A_ij, V picks up B_jk.
    #    For schoolbook term A_ij B_jk:
    #    U[i,j] = 1
    #    V[j,k] = 1
    #    W[k,i] = 1
    #    
    #    Terms:
    #    1: A00 B0k -> C0k. U[0,0]=1, V[0,k]=1, W[k,0]=1
    #    2: A01 B1k -> C0k. U[0,1]=1, V[1,k]=1, W[k,0]=1
    #    3: A10 B0k -> C1k. U[1,0]=1, V[0,k]=1, W[k,1]=1
    #    4: A11 B1k -> C1k. U[1,1]=1, V[1,k]=1, W[k,1]=1
    #    
    #    Note: In the global V matrix, the row index is j*p + k.
    #    Since this is the last block, k=14.
    #    V[0, 14] corresponds to row 0*15+14 = 14.
    #    V[1, 14] corresponds to row 1*15+14 = 29.
    
    # We have 7 Strassen blocks (k=0..13) and 1 Schoolbook block (k=14).
    
    for block_idx in range(7):
        k_start = block_idx * 2
        
        if time.time() > deadline:
            # Return current best effort (which is incomplete if timeout hits, but construction is fast)
            # For robustness, we just return the partial matrices if we hit deadline, 
            # though it's highly unlikely for this static construction.
            pass 

        # Fill U
        for u_row in range(4):
            for c in range(7):
                U[u_row][col_offset + c] = U_strassen[u_row][c]
        
        # Fill V
        for v_row in range(4):
            j = v_row // 2
            k_local = v_row % 2
            k_global = k_start + k_local
            v_idx = j * p + k_global
            for c in range(7):
                V[v_idx][col_offset + c] = V_strassen[v_row][c]
                
        # Fill W
        for w_row in range(4):
            k_local = w_row // 2
            i = w_row % 2
            k_global = k_start + k_local
            w_idx = k_global * n + i
            for c in range(7):
                W[w_idx][col_offset + c] = W_strassen[w_row][c]
                
        col_offset += 7
        
    # Schoolbook block at k=14
    k_start = 14
    
    # U is identity for the 4 rows (A00, A01, A10, A11)
    U[0][col_offset] = 1
    U[1][col_offset+1] = 1
    U[2][col_offset+2] = 1
    U[3][col_offset+3] = 1
    
    # V
    # Term 1: A00 * B0,14. V[0, 14] = 1. Row index in V: 0*15 + 14 = 14.
    V[14][col_offset] = 1
    # Term 2: A01 * B1,14. V[1, 14] = 1. Row index in V: 1*15 + 14 = 29.
    V[29][col_offset+1] = 1
    # Term 3: A10 * B0,14. V[0, 14] = 1. Row index in V: 0*15 + 14 = 14.
    V[14][col_offset+2] = 1
    # Term 4: A11 * B1,14. V[1, 14] = 1. Row index in V: 1*15 + 14 = 29.
    V[29][col_offset+3] = 1
    
    # W
    # Term 1: -> C0,14. W[14, 0] = 1. Row index in W: 14*2 + 0 = 28.
    W[28][col_offset] = 1
    # Term 2: -> C0,14. W[14, 0] = 1. Row index in W: 14*2 + 0 = 28.
    W[28][col_offset+1] = 1
    # Term 3: -> C1,14. W[14, 1] = 1. Row index in W: 14*2 + 1 = 29.
    W[29][col_offset+2] = 1
    # Term 4: -> C1,14. W[14, 1] = 1. Row index in W: 14*2 + 1 = 29.
    W[29][col_offset+3] = 1
    
    return {"U": U, "V": V, "W": W}
