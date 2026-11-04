import numpy as np

def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 4, 5
    
    # Total rank
    # 4 Strassen blocks (rank 7 each)
    # 2 Schoolbook 2x2x1 blocks (rank 4 each)
    total_rank = 4 * 7 + 2 * 4
    # 28 + 8 = 36
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Define Strassen factors (local 0-1 indices)
    # U_strassen[4, 7], V_strassen[4, 7], W_strassen[4, 7]
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Define Schoolbook 2x2x1 factors (local 0-1 indices for i,j; 0 for k)
    # U_sb[4, 4], V_sb[2, 4], W_sb[2, 4]
    U_sb = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    V_sb = [
        [1, 0, 1, 0], # e (B0)
        [0, 1, 0, 1]  # f (B1)
    ]
    W_sb = [
        [1, 1, 0, 0], # C0
        [0, 0, 1, 1]  # C1
    ]
    
    # Iterate over blocks
    # block_m: 0 (j=0,1), 1 (j=2,3)
    # block_p: 0 (k=0,1, Type S), 1 (k=2,3, Type S), 2 (k=4, Type L)
    
    for block_m in range(2):
        for block_p in range(3):
            # Determine type and global index offsets
            j_offset = block_m * 2
            if block_p < 2:
                k_offset = block_p * 2
                type_str = 'S'
            else:
                k_offset = 4
                type_str = 'L'
                
            if type_str == 'S':
                # 7 columns
                for c in range(7):
                    # Fill U
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = i_local
                            j_global = j_local + j_offset
                            u_val = U_strassen[i_local * 2 + j_local][c]
                            if u_val != 0:
                                U[i_global * m + j_global][col_idx + c] = u_val
                                
                    # Fill V
                    for j_local in range(2):
                        for k_local in range(2):
                            j_global = j_local + j_offset
                            k_global = k_local + k_offset
                            v_val = V_strassen[j_local * 2 + k_local][c]
                            if v_val != 0:
                                V[j_global * p + k_global][col_idx + c] = v_val
                                
                    # Fill W
                    for k_local in range(2):
                        for i_local in range(2):
                            k_global = k_local + k_offset
                            i_global = i_local
                            w_val = W_strassen[k_local * n + i_local][c]
                            if w_val != 0:
                                W[k_global * n + i_global][col_idx + c] = w_val
            else:
                # 4 columns
                for c in range(4):
                    # Fill U
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = i_local
                            j_global = j_local + j_offset
                            u_val = U_sb[i_local * 2 + j_local][c]
                            if u_val != 0:
                                U[i_global * m + j_global][col_idx + c] = u_val
                                
                    # Fill V
                    for j_local in range(2):
                        k_local = 0
                        j_global = j_local + j_offset
                        k_global = k_local + k_offset
                        v_val = V_sb[j_local * 1 + k_local][c] # V_sb is 2x4, index j_local*1+k_local
                        if v_val != 0:
                            V[j_global * p + k_global][col_idx + c] = v_val
                            
                    # Fill W
                    for i_local in range(2):
                        k_local = 0
                        k_global = k_local + k_offset
                        i_global = i_local
                        w_val = W_sb[k_local * n + i_local][c] # W_sb is 2x4, index k_local*n+i_local
                        if w_val != 0:
                            W[k_global * n + i_global][col_idx + c] = w_val
                            
            col_idx += (7 if type_str == 'S' else 4)
            
    # Verify
    assert _validate(U, V, W, n, m, p), "Decomposition failed validation!"
    
    return {"U": U, "V": V, "W": W}

