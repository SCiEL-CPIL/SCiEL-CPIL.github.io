import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find a better decomposition for M_3,3,3
    # If found, extend to M_3,3,4
    # Target rank for M_3,3,3: 26
    # Total rank for M_3,3,4: 26 + 9 = 35
    
    target_rank_333 = 26
    extra_rank = 9
    target_total_rank = target_rank_333 + extra_rank
    
    if target_total_rank < best_rank:
        # Search for M_3,3,3 decomposition
        # Tensor T_333[i, j, k] = 1 if i,j,k are indices of A, B, C
        # We need U (9 x r), V (9 x r), W (9 x r)
        # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_333[i, j, k]
        
        # We will use a simple random search with local improvements
        # Or just random restarts
        
        found = False
        for attempt in range(50): # Max 50 attempts
            if time.time() > deadline:
                break
            
            r = target_rank_333
            U = np.zeros((n*m, r))
            V = np.zeros((m*3, r)) # m*p where p=3
            W = np.zeros((3*n, r)) # p*n where p=3
            
            # Initialize with random integers from {-1, 0, 1}
            U = np.random.randint(-1, 2, (n*m, r))
            V = np.random.randint(-1, 2, (m*3, r))
            W = np.random.randint(-1, 2, (3*n, r))
            
            # Check if this decomposition is exact
            # Compute the tensor from U, V, W
            T_comp = np.zeros((n, m, 3))
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        s = 0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j*3+k, r_idx] * W[k*n+i, r_idx]
                        T_comp[i, j, k] = s
                        
            # The target tensor is 1 if i,j,k are valid indices (which they always are in schoolbook)
            # Wait, the tensor T_333[i, j, k] is 1 for all i, j, k?
            # Yes, because C_ik = sum_j A_ij B_jk.
            # The coefficient of A_ij B_jk is 1.
            # So T_333[i, j, k] = 1 for all i, j, k.
            
            if np.all(T_comp == 1):
                found = True
                break
        
        if found:
            # Construct M_3,3,4 solution
            # U_334 is same as U_333 for the first 26 cols
            # We need to add 9 cols for the 4th column of B
            # The 4th column computation: C_i3 = sum_j A_ij B_j3
            # This requires 9 terms: A_ij * B_j3
            # So we add 9 columns to U, V, W
            
            total_rank = target_total_rank
            U_334 = np.zeros((n*m, total_rank))
            V_334 = np.zeros((m*4, total_rank))
            W_334 = np.zeros((4*n, total_rank))
            
            # Copy the first 26 cols
            U_334[:, :26] = U
            V_334[:m*3, :26] = V
            W_334[:3*n, :26] = W
            
            # Add the 9 schoolbook terms for the 4th column
            # Indices for 4th column of B: j*4 + 3
            # Indices for 4th column of C: 3*n + i
            for j in range(m):
                for i in range(n):
                    col_idx = 26 + j*n + i
                    U_334[i*m + j, col_idx] = 1
                    V_334[j*4 + 3, col_idx] = 1
                    W_334[3*n + i, col_idx] = 1
            
            # Convert to lists of lists
            U_list = U_334.tolist()
            V_list = V_334.tolist()
            W_list = W_334.tolist()
            
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return best_result
