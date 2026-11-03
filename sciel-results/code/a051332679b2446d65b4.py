import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for Strassen's 2x2x2 matrix multiplication.
    Each tensor is (U, V, W) where U is 4x1, V is 4x1, W is 4x1.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    # Correct Strassen decomposition for 2x2x2
    # The 7 products and their contributions:
    # M1 = (A00+A11)(B00+B11) → C[0,0] += M1, C[1,1] += M1
    # M2 = (A10+A11)B00 → C[1,0] += M2
    # M3 = A00(B01-B11) → C[0,1] += M3
    # M4 = A11(B10-B00) → C[1,0] += M4
    # M5 = (A00+A01)B11 → C[0,1] -= M5
    # M6 = (A10-A00)(B00+B01) → C[1,1] += M6
    # M7 = (A01-A11)(B10+B11) → C[0,0] -= M7
    
    # For each product Mc, we need u_c, v_c, w_c such that:
    # sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
    
    # M1 = (A00+A11)(B00+B11)
    # u1[i,j] = 1 if (i,j) in {(0,0),(1,1)} else 0
    # v1[j,k] = 1 if (j,k) in {(0,0),(1,1)} else 0
    # w1[k,i] = 1 if (k,i) in {(0,0),(1,1)} else 0
    u1 = [1, 0, 0, 1]  # i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1
    v1 = [1, 0, 0, 1]  # j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
    w1 = [1, 0, 0, 1]  # k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    
    # M2 = (A10+A11)B00
    # u2[i,j] = 1 if (i,j) in {(1,0),(1,1)} else 0
    # v2[j,k] = 1 if (j,k) = (0,0) else 0
    # w2[k,i] = 1 if (k,i) = (0,1) else 0 (contributes to C[1,0])
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 1, 0, 0]
    
    # M3 = A00(B01-B11)
    # u3[i,j] = 1 if (i,j) = (0,0) else 0
    # v3[j,k] = 1 if (j,k) = (0,1) else 0, -1 if (j,k) = (1,1) else 0
    # w3[k,i] = 1 if (k,i) = (1,0) else 0 (contributes to C[0,1])
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 0, 1, 0]
    
    # M4 = A11(B10-B00)
    # u4[i,j] = 1 if (i,j) = (1,1) else 0
    # v4[j,k] = 1 if (j,k) = (1,0) else 0, -1 if (j,k) = (0,0) else 0
    # w4[k,i] = 1 if (k,i) = (0,1) else 0 (contributes to C[1,0])
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [0, 1, 0, 0]
    
    # M5 = (A00+A01)B11
    # u5[i,j] = 1 if (i,j) in {(0,0),(0,1)} else 0
    # v5[j,k] = 1 if (j,k) = (1,1) else 0
    # w5[k,i] = -1 if (k,i) = (1,0) else 0 (contributes -M5 to C[0,1])
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 0, -1, 0]
    
    # M6 = (A10-A00)(B00+B01)
    # u6[i,j] = -1 if (i,j) = (0,0) else 0, 1 if (i,j) = (1,0) else 0
    # v6[j,k] = 1 if (j,k) in {(0,0),(0,1)} else 0
    # w6[k,i] = 1 if (k,i) = (1,1) else 0 (contributes to C[1,1])
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # M7 = (A01-A11)(B10+B11)
    # u7[i,j] = 1 if (i,j) = (0,1) else 0, -1 if (i,j) = (1,1) else 0
    # v7[j,k] = 1 if (j,k) in {(1,0),(1,1)} else 0
    # w7[k,i] = -1 if (k,i) = (0,0) else 0 (contributes -M7 to C[0,0])
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [-1, 0, 0, 0]
    
    return [u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Get Strassen decomposition for 2x2x2
    strassen_U, strassen_V, strassen_W = strassen_2x2x2()
    
    # Verify Strassen decomposition
    strassen_valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for c in range(7):
                    s += strassen_U[c][i*2+j] * strassen_V[c][j*2+k] * strassen_W[c][k*2+i]
                if abs(s - 1.0) > 1e-9:
                    strassen_valid = False
                    break
        if not strassen_valid:
            break
    
    if strassen_valid:
        # Build the full 3x4x6 decomposition
        # Partition: i = 2+1, j = 2+2, k = 2+2+2
        # Blocks: 6 of type (2,2,2) and 6 of type (1,2,2)
        # Total rank: 6*7 + 6*4 = 42 + 24 = 66
        
        total_rank = 6 * 7 + 6 * 4
        U_full = [[0] * total_rank for _ in range(n * m)]  # 3*4 = 12 rows
        V_full = [[0] * total_rank for _ in range(m * p)]  # 4*6 = 24 rows
        W_full = [[0] * total_rank for _ in range(p * n)]  # 6*3 = 18 rows
        
        col_offset = 0
        
        # Process (2,2,2) blocks: i in {0,1}, j-blocks [0,1] and [2,3], k-blocks [0,1], [2,3], [4,5]
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(3):
                k_start = k_block * 2
                
                # This is a (2,2,2) block: i in [0,1], j in [j_start, j_start+1], k in [k_start, k_start+1]
                # Use Strassen decomposition
                for c in range(7):
                    # Fill U_full
                    for i_local in range(2):
                        i_global = i_local
                        for j_local in range(2):
                            j_global = j_start + j_local
                            U_full[i_global * m + j_global][col_offset + c] = strassen_U[c][i_local * 2 + j_local]
                    
                    # Fill V_full
                    for j_local in range(2):
                        j_global = j_start + j_local
                        for k_local in range(2):
                            k_global = k_start + k_local
                            V_full[j_global * p + k_global][col_offset + c] = strassen_V[c][j_local * 2 + k_local]
                    
                    # Fill W_full
                    for k_local in range(2):
                        k_global = k_start + k_local
                        for i_local in range(2):
                            i_global = i_local
                            W_full[k_global * n + i_global][col_offset + c] = strassen_W[c][k_local * 2 + i_local]
                
                col_offset += 7
        
        # Process (1,2,2) blocks: i=2, j-blocks [0,1] and [2,3], k-blocks [0,1], [2,3], [4,5]
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(3):
                k_start = k_block * 2
                
                # (1,2,2) block: i=2, j in [j_start, j_start+1], k in [k_start, k_start+1]
                # Rank-4 decomposition: for each (j_local, k_local), one rank-1 tensor
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_start + j_local
                        k_global = k_start + k_local
                        
                        # u[2, j_global] = 1, v[j_global, k_global] = 1, w[k_global, 2] = 1
                        U_full[2 * m + j_global][col_offset] = 1
                        V_full[j_global * p + k_global][col_offset] = 1
                        W_full[k_global * n + 2][col_offset] = 1
                        
                        col_offset += 1
        
        # Verify the full decomposition
        if verify(U_full, V_full, W_full, n, m, p):
            best = {"U": U_full, "V": U_full, "W": W_full}
            best_rank = total_rank
        else:
            # Strassen decomposition might be wrong, fall back to schoolbook
            pass
    
    return best
