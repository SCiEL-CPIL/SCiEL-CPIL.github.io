"""Matrix multiplication 3x4x6 with optimized block decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1
        
        # M3 = A00(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1
        
        # M4 = A11(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1
        W_s[2, 3] = 1
        
        # M5 = (A00+A10)B11
        U_s[0, 4] = 1; U_s[2, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1
        W_s[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[2, 5] = 1
        W_s[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[1, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s
    
    # Schoolbook decomposition for 1x2x2
    def schoolbook_1x2x2():
        U_s = np.zeros((2, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((2, 4))
        
        col = 0
        for d in range(2):
            for c in range(2):
                U_s[d, col] = 1
                V_s[d*2+c, col] = 1
                W_s[c, col] = 1
                col += 1
        
        return U_s, V_s, W_s

    # Rank of 1x1x2 (single element times 1x2): rank 2
    def schoolbook_1x1x2():
        U_s = np.zeros((1, 2))
        V_s = np.zeros((2, 2))
        W_s = np.zeros((2, 2))
        U_s[0, 0] = 1; U_s[0, 1] = 1
        V_s[0, 0] = 1; V_s[1, 1] = 1
        W_s[0, 0] = 1; W_s[1, 1] = 1
        return U_s, V_s, W_s

    # Rank of 2x1x2: rank 4 (schoolbook)
    def schoolbook_2x1x2():
        U_s = np.zeros((2, 4))
        V_s = np.zeros((2, 4))
        W_s = np.zeros((2, 4))
        col = 0
        for a in range(2):
            for c in range(2):
                U_s[a, col] = 1
                V_s[c, col] = 1
                W_s[c, col] = 1
                col += 1
        return U_s, V_s, W_s

    # Schoolbook for general a x 2 x b
    def schoolbook_a2b(a, b):
        rank = a * 2 * b
        U_s = np.zeros((a * 2, rank))
        V_s = np.zeros((2 * b, rank))
        W_s = np.zeros((b * a, rank))
        col = 0
        for i in range(a):
            for j in range(2):
                for k in range(b):
                    U_s[i*2+j, col] = 1
                    V_s[j*b+k, col] = 1
                    W_s[k*a+i, col] = 1
                    col += 1
        return U_s, V_s, W_s

    # Build decomposition given a partition of A rows and B cols
    def build_decomp(a_splits, b_splits, mid_split=[2,2]):
        """
        a_splits: list of sizes summing to n=3
        b_splits: list of sizes summing to p=6
        mid_split: how to split the middle dim (4)
        For each (a_block, b_block, mid_block), use best known decomposition.
        """
        all_U = []
        all_V = []
        all_W = []
        
        def add_block(U_block, V_block, W_block):
            all_U.append(U_block)
            all_V.append(V_block)
            all_W.append(W_block)
        
        a_offset = 0
        for a_size in a_splits:
            b_offset = 0
            for b_size in b_splits:
                mid_offset = 0
                for mid_size in mid_split:
                    # Sub-subproblem: a_size x mid_size x b_size
                    if a_size == 2 and mid_size == 2 and b_size == 2:
                        U_s, V_s, W_s = strassen_2x2x2()
                        U_g = np.zeros((12, 7))
                        V_g = np.zeros((24, 7))
                        W_g = np.zeros((18, 7))
                        for a in range(2):
                            for d in range(2):
                                local_row = a * 2 + d
                                global_i = a_offset + a
                                global_j = mid_offset + d
                                U_g[global_i * 4 + global_j, :] = U_s[local_row, :]
                        for d in range(2):
                            for c in range(2):
                                local_row = d * 2 + c
                                global_j = mid_offset + d
                                global_k = b_offset + c
                                V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
                        for c in range(2):
                            for a in range(2):
                                local_row = c * 2 + a
                                global_k = b_offset + c
                                global_i = a_offset + a
                                W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and mid_size == 2 and b_size == 2:
                        U_s, V_s, W_s = schoolbook_1x2x2()
                        U_g = np.zeros((12, 4))
                        V_g = np.zeros((24, 4))
                        W_g = np.zeros((18, 4))
                        for d in range(2):
                            global_i = a_offset
                            global_j = mid_offset + d
                            U_g[global_i * 4 + global_j, :] = U_s[d, :]
                        for d in range(2):
                            for c in range(2):
                                local_row = d * 2 + c
                                global_j = mid_offset + d
                                global_k = b_offset + c
                                V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
                        for c in range(2):
                            local_row = c
                            global_k = b_offset + c
                            global_i = a_offset
                            W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 2 and mid_size == 1 and b_size == 2:
                        # 2x1x2 schoolbook, rank 4
                        U_s, V_s, W_s = schoolbook_2x1x2()
                        U_g = np.zeros((12, 4))
                        V_g = np.zeros((24, 4))
                        W_g = np.zeros((18, 4))
                        for a in range(2):
                            for c in range(2):
                                global_i = a_offset + a
                                global_j = mid_offset
                                U_g[global_i * 4 + global_j, c] = U_s[a, c]
                        for c in range(2):
                            global_j = mid_offset
                            global_k = b_offset + c
                            V_g[global_j * 6 + global_k, :] = V_s[:, c]
                        for a in range(2):
                            for c in range(2):
                                global_k = b_offset + c
                                global_i = a_offset + a
                                W_g[global_k * 3 + global_i, c] = W_s[a, c]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and mid_size == 1 and b_size == 2:
                        # 1x1x2, rank 2
                        U_s, V_s, W_s = schoolbook_1x1x2()
                        U_g = np.zeros((12, 2))
                        V_g = np.zeros((24, 2))
                        W_g = np.zeros((18, 2))
                        for c in range(2):
                            global_i = a_offset
                            global_j = mid_offset
                            U_g[global_i * 4 + global_j, c] = U_s[0, c]
                            global_k = b_offset + c
                            V_g[global_j * 6 + global_k, c] = V_s[c, c]
                            W_g[global_k * 3 + global_i, c] = W_s[c, c]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 2 and mid_size == 2 and b_size == 1:
                        # 2x2x1, rank 4
                        U_s, V_s, W_s = schoolbook_a2b(2, 1)
                        U_g = np.zeros((12, 4))
                        V_g = np.zeros((24, 4))
                        W_g = np.zeros((18, 4))
                        for a in range(2):
                            for d in range(2):
                                global_i = a_offset + a
                                global_j = mid_offset + d
                                U_g[global_i * 4 + global_j, :] = U_s[a*2+d, :]
                        for d in range(2):
                            global_j = mid_offset + d
                            global_k = b_offset
                            V_g[global_j * 6 + global_k, :] = V_s[d, :]
                        for a in range(2):
                            global_i = a_offset + a
                            global_k = b_offset
                            W_g[global_k * 3 + global_i, :] = W_s[a, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and mid_size == 2 and b_size == 1:
                        # 1x2x1, rank 2
                        U_s, V_s, W_s = schoolbook_a2b(1, 1)
                        U_g = np.zeros((12, 2))
                        V_g = np.zeros((24, 2))
                        W_g = np.zeros((18, 2))
                        for d in range(2):
                            global_i = a_offset
                            global_j = mid_offset + d
                            U_g[global_i * 4 + global_j, :] = U_s[d, :]
                        for d in range(2):
                            global_j = mid_offset + d
                            global_k = b_offset
                            V_g[global_j * 6 + global_k, :] = V_s[d, :]
                        global_i = a_offset
                        global_k = b_offset
                        W_g[global_k * 3 + global_i, :] = W_s[0, :]
                        add_block(U_g, V_g, W_g)
                    elif a_size == 1 and mid_size == 1 and b_size == 1:
                        # 1x1x1, rank 1
                        U_g = np.zeros((12, 1))
                        V_g = np.zeros((24, 1))
                        W_g = np.zeros((18, 1))
                        U_g[a_offset * 4 + mid_offset, 0] = 1
                        V_g[mid_offset * 6 + b_offset, 0] = 1
                        W_g[b_offset * 3 + a_offset, 0] = 1
                        add_block(U_g, V_g, W_g)
                    else:
                        # General fallback: schoolbook
                        rank = a_size * mid_size * b_size
                        U_g = np.zeros((12, rank))
                        V_g = np.zeros((24, rank))
                        W_g = np.zeros((18, rank))
                        col = 0
                        for a in range(a_size):
                            for d in range(mid_size):
                                for c in range(b_size):
                                    U_g[(a_offset+a)*4 + (mid_offset+d), col] = 1
                                    V_g[(mid_offset+d)*6 + (b_offset+c), col] = 1
                                    W_g[(b_offset+c)*3 + (a_offset+a), col] = 1
                                    col += 1
                        add_block(U_g, V_g, W_g)
                    mid_offset += mid_size
            b_offset += b_size
        a_offset += a_size
        
        total_rank = sum(U.shape[1] for U in all_U)
        U_final = np.zeros((12, total_rank))
        V_final = np.zeros((24, total_rank))
        W_final = np.zeros((18, total_rank))
        col_offset = 0
        for U_b, V_b, W_b in zip(all_U, all_V, all_W):
            r = U_b.shape[1]
            U_final[:, col_offset:col_offset+r] = U_b
            V_final[:, col_offset:col_offset+r] = V_b
            W_final[:, col_offset:col_offset+r] = W_b
            col_offset += r
        return U_final, V_final, W_final, total_rank

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Search over different partitions
    best_rank = float('inf')
    best_result = None
    
    # Possible A row splits (must sum to 3)
    a_split_options = [[2,1], [1,2], [1,1,1], [3]]
    # Possible B col splits (must sum to 6)
    b_split_options = [[2,2,2], [3,3], [2,4], [4,2], [1,2,3], [3,2,1], [1,1,4], [1,4,1], [4,1,1], [2,1,3], [3,1,2], [1,3,2], [2,3,1], [6], [1,5], [5,1], [1,1,1,3], [1,1,3,1], [1,3,1,1], [3,1,1,1], [1,1,1,1,2], [1,1,1,2,1], [1,1,2,1,1], [1,2,1,1,1], [2,1,1,1,1]]
    # Possible middle splits (must sum to 4)
    mid_split_options = [[2,2], [1,3], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1], [4], [2,1,1], [1,2,1]]
    
    for a_splits in a_split_options:
        if sum(a_splits) != n:
            continue
        for b_splits in b_split_options:
            if sum(b_splits) != p:
                continue
            for mid_splits in mid_split_options:
                if sum(mid_splits) != m:
                    continue
                if time.time() - start_time > DEADLINE - 10:
                    break
                try:
                    U, V, W, rank = build_decomp(a_splits, b_splits, mid_splits)
                    if check_decomposition(U, V, W):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = (U, V, W)
                except Exception:
                    pass
    
    if best_result is not None:
        U_final, V_final, W_final = best_result
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fb = np.zeros((n*m, rank))
    V_fb = np.zeros((m*p, rank))
    W_fb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
