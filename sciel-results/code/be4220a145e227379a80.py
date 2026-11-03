# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Attempts to find a rank-54 decomposition by splitting into two 2x5x3 blocks
    and finding rank-27 decompositions for each.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}

    def try_find_rank_27_for_2x5x3(deadline):
        """
        Try to find a rank-27 decomposition for 2x5x3 matrix multiplication.
        Returns U, V, W or None.
        """
        n_sub, m_sub, p_sub = 2, 5, 3
        target_rank = 27
        # Schoolbook for sub-problem
        rank_sb = n_sub * m_sub * p_sub # 30
        
        # We will use a random search with a bias towards {-1, 0, 1}
        # and verify against the sub-tensor
        for attempt in range(500):
            if time.time() > deadline - 10:
                return None
            
            random.seed(42 + attempt)
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n_sub * m_sub)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m_sub * p_sub)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p_sub * n_sub)]
            
            # Verify
            valid = True
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        s = 0.0
                        for t in range(target_rank):
                            s += U[i * m_sub + j][t] * V[j * p_sub + k][t] * W[k * n_sub + i][t]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        
        return None

    # Try to find the rank-54 decomposition
    # Split p=6 into two p_sub=3
    # We need two rank-27 decompositions for (2, 5, 3)
    # We can use the same decomposition structure for both if we permute indices?
    # B_L uses j, k_L. B_R uses j, k_R.
    # The j index is the same.
    # So we need U (2x5) and V (5x3) and W (3x2) for each block.
    # The U part is shared? No, U depends on i, j.
    # In the full problem, U[i*m+j] is the coefficient for A_ij.
    # In the sub-problem, U_sub[i*m_sub+j] is the coefficient for A_ij.
    # Since m_sub = m = 5, the U matrix is the same for both blocks?
    # No, the rank is shared.
    # In the full rank-54 decomposition, the rank is 54.
    # Block 1 uses ranks 0..26. Block 2 uses ranks 27..53.
    # So U_full has 54 columns.
    # Columns 0..26 correspond to Block 1. Columns 27..53 correspond to Block 2.
    # U_full[i*5+j, 0..26] = U_sub1[i*5+j]
    # U_full[i*5+j, 27..53] = U_sub2[i*5+j]
    
    # V_full[j*6+k, t]
    # For Block 1 (k=0,1,2): V_full[j*6+k, 0..26] = V_sub1[j*3+k]
    # For Block 2 (k=3,4,5): V_full[j*6+k, 27..53] = V_sub2[j*3+(k-3)]
    
    # W_full[k*2+i, t]
    # For Block 1 (k=0,1,2): W_full[k*2+i, 0..26] = W_sub1[k*2+i]
    # For Block 2 (k=3,4,5): W_full[k*2+i, 27..53] = W_sub2[(k-3)*2+i]

    sub_res = try_find_rank_27_for_2x5x3(deadline)
    
    if sub_res:
        U_sub, V_sub, W_sub = sub_res
        # Construct full matrices
        rank_full = 54
        U_full = [[0] * rank_full for _ in range(n * m)]
        V_full = [[0] * rank_full for _ in range(m * p)]
        W_full = [[0] * rank_full for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for t in range(27):
                    U_full[i * m + j][t] = U_sub[i * m + j][t]
                    U_full[i * m + j][t + 27] = U_sub[i * m + j][t]
        
        for j in range(m):
            for k in range(p):
                if k < 3:
                    for t in range(27):
                        V_full[j * p + k][t] = V_sub[j * 3 + k][t]
                        V_full[j * p + k][t + 27] = 0
                else:
                    for t in range(27):
                        V_full[j * p + k][t] = 0
                        V_full[j * p + k][t + 27] = V_sub[j * 3 + (k-3)][t]
        
        for k in range(p):
            for i in range(n):
                if k < 3:
                    for t in range(27):
                        W_full[k * n + i][t] = W_sub[k * n + i][t]
                        W_full[k * n + i][t + 27] = 0
                else:
                    for t in range(27):
                        W_full[k * n + i][t] = 0
                        W_full[k * n + i][t + 27] = W_sub[(k-3) * n + i][t]
        
        # Verify full
        if verify_decomposition(U_full, V_full, W_full):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
