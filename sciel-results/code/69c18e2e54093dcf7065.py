import time
import numpy as np
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # The tensor identity requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    def try_find_decomposition(rank, num_trials=5000):
        """
        Attempts to find a rank decomposition by fixing U and V randomly
        and solving for W linearly.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize U and V with sparse random integers
            # We want U and V to be sparse to make W potentially simple
            # Each column of U has n*m entries. We pick a few to be non-zero.
            # Similarly for V.
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # Strategy: For each rank r, pick a random subset of indices for U and V
            # to be non-zero.
            for r in range(rank):
                # Pick 2 or 3 non-zeros in U[:, r]
                num_nonzero_u = random.randint(2, 4)
                idx_u = random.sample(range(n * m), num_nonzero_u)
                for idx in idx_u:
                    U[idx, r] = random.choice([-1, 1])
                
                # Pick 2 or 3 non-zeros in V[:, r]
                num_nonzero_v = random.randint(2, 4)
                idx_v = random.sample(range(m * p), num_nonzero_v)
                for idx in idx_v:
                    V[idx, r] = random.choice([-1, 1])
            
            # Now solve for W
            # For each (i, k), we have m equations (one for each j)
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_ik be a m x rank matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then A_ik @ W_col_ik = 1 (vector of ones)
            
            W = np.zeros((p * n, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct the m x rank matrix
                    A_ik = np.zeros((m, rank))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik @ W_col = ones(m)
                    # W_col is a vector of length rank
                    try:
                        # Use least squares or direct solve
                        # Since it's underdetermined (rank > m usually), we can use pinv
                        # Or we can look for a sparse solution
                        W_col, residuals, rank_lsq, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        
                        # Check if the solution is "close" to integers/half-integers
                        # and if the residual is small
                        if residuals.size > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                        
                        # Check if W_col entries are nice
                        # We allow integers and half-integers
                        w_nice = np.all(np.abs(W_col * 2 - np.round(W_col * 2)) < 1e-9)
                        if not w_nice:
                            # Try to find a sparse solution if possible?
                            # For now, reject if not nice
                            valid = False
                            break
                            
                        W[k * n + i, :] = W_col
                        
                    except Exception:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Final verification
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify with random matrices
                A_test = np.random.randint(-5, 5, (n, m))
                B_test = np.random.randint(-5, 5, (m, p))
                C_true = A_test @ B_test
                
                C_calc = np.zeros((n, p))
                for r in range(rank):
                    a_proj = np.zeros(n)
                    for i in range(n):
                        for j in range(m):
                            a_proj[i] += A_test[i, j] * U_list[i * m + j][r]
                    b_proj = np.zeros(p)
                    for k in range(p):
                        for j in range(m):
                            b_proj[k] += B_test[j, k] * V_list[j * p + k][r]
                    for i in range(n):
                        for k in range(p):
                            C_calc[i, k] += W_list[k * n + i][r] * a_proj[i] * b_proj[k]
                
                if np.allclose(C_calc, C_true, atol=1e-9):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def strassen_fallback():
        # The known rank-28 solution from the seed program
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        def add_strassen(col_offset, A_block, B_block, C_block_W_indices):
            a00_idx = A_block[(0,0)]
            a01_idx = A_block[(0,1)]
            a10_idx = A_block[(1,0)]
            a11_idx = A_block[(1,1)]
            
            b00_idx = B_block[(0,0)]
            b01_idx = B_block[(0,1)]
            b10_idx = B_block[(1,0)]
            b11_idx = B_block[(1,1)]
            
            c00_w = C_block_W_indices[(0,0)]
            c01_w = C_block_W_indices[(0,1)]
            c10_w = C_block_W_indices[(1,0)]
            c11_w = C_block_W_indices[(1,1)]
            
            U[a00_idx][col_offset] += 1.0
            U[a11_idx][col_offset] += 1.0
            V[b00_idx][col_offset] += 1.0
            V[b11_idx][col_offset] += 1.0
            W[c00_w][col_offset] += 1.0
            W[c11_w][col_offset] += 1.0
            
            U[a10_idx][col_offset+1] += 1.0
            U[a11_idx][col_offset+1] += 1.0
            V[b00_idx][col_offset+1] += 1.0
            W[c10_w][col_offset+1] += 1.0
            
            U[a00_idx][col_offset+2] += 1.0
            V[b10_idx][col_offset+2] += 1.0
            V[b11_idx][col_offset+2] += -1.0
            W[c01_w][col_offset+2] += 1.0
            W[c11_w][col_offset+2] += 1.0
            
            U[a11_idx][col_offset+3] += 1.0
            V[b01_idx][col_offset+3] += 1.0
            V[b00_idx][col_offset+3] += -1.0
            W[c00_w][col_offset+3] += 1.0
            W[c10_w][col_offset+3] += 1.0
            
            U[a00_idx][col_offset+4] += 1.0
            U[a01_idx][col_offset+4] += 1.0
            V[b11_idx][col_offset+4] += 1.0
            W[c00_w][col_offset+4] += -1.0
            W[c01_w][col_offset+4] += 1.0
            
            U[a10_idx][col_offset+5] += 1.0
            U[a00_idx][col_offset+5] += -1.0
            V[b00_idx][col_offset+5] += 1.0
            V[b01_idx][col_offset+5] += 1.0
            W[c11_w][col_offset+5] += 1.0
            
            U[a01_idx][col_offset+6] += 1.0
            U[a11_idx][col_offset+6] += -1.0
            V[b10_idx][col_offset+6] += 1.0
            V[b11_idx][col_offset+6] += 1.0
            W[c00_w][col_offset+6] += 1.0
            W[c11_w][col_offset+6] += -1.0
            
            return col_offset + 7
        
        A1 = {(0,0): 0, (0,1): 1, (1,0): 2, (1,1): 3}
        A2 = {(0,0): 4, (0,1): 5, (1,0): 6, (1,1): 7}
        
        B11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
        B12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
        B21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
        B22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
        
        C1_W = {(0,0): 0, (0,1): 2, (1,0): 1, (1,1): 3}
        C2_W = {(0,0): 4, (0,1): 6, (1,0): 5, (1,1): 7}
        
        col = 0
        col = add_strassen(col, A1, B11, C1_W)
        col = add_strassen(col, A2, B21, C1_W)
        col = add_strassen(col, A1, B12, C2_W)
        col = add_strassen(col, A2, B22, C2_W)
        
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank decomposition
    # Known lower bounds suggest rank 20 might be possible
    for rank in [20, 22, 24, 26]:
        if time.time() > deadline:
            break
        sol = try_find_decomposition(rank, num_trials=2000)
        if sol:
            return sol
            
    # Fallback to rank 28
    return strassen_fallback()
# EVOLVE-BLOCK-END
