import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 145

    # Pre-compute Strassen blocks
    def get_strassen_block(k_start):
        U_block = np.zeros((4, 7))
        V_block = np.zeros((14, 7))
        W_block = np.zeros((14, 7))
        
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_V_local = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        strassen_W_local = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]
        
        for r in range(7):
            for i in range(4):
                U_block[i, r] = strassen_U[r][i]
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0: v_idx = 0 * p + (2 * k_start)
                elif loc_idx == 1: v_idx = 0 * p + (2 * k_start + 1)
                elif loc_idx == 2: v_idx = 1 * p + (2 * k_start)
                elif loc_idx == 3: v_idx = 1 * p + (2 * k_start + 1)
                V_block[v_idx, r] = coeff
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0: w_idx = (2 * k_start) * n + 0
                elif loc_idx == 1: w_idx = (2 * k_start + 1) * n + 0
                elif loc_idx == 2: w_idx = (2 * k_start) * n + 1
                elif loc_idx == 3: w_idx = (2 * k_start + 1) * n + 1
                W_block[w_idx, r] = coeff
        return U_block, V_block, W_block

    # Pre-compute Rank 25 solution
    U_25 = np.zeros((4, 25))
    V_25 = np.zeros((14, 25))
    W_25 = np.zeros((14, 25))
    
    U_b1, V_b1, W_b1 = get_strassen_block(0)
    U_25[:, 0:7] += U_b1
    V_25[:, 0:7] += V_b1
    W_25[:, 0:7] += W_b1
    
    U_b2, V_b2, W_b2 = get_strassen_block(1)
    U_25[:, 7:14] += U_b2
    V_25[:, 7:14] += V_b2
    W_25[:, 7:14] += W_b2
    
    U_b3, V_b3, W_b3 = get_strassen_block(2)
    U_25[:, 14:21] += U_b3
    V_25[:, 14:21] += V_b3
    W_25[:, 14:21] += W_b3
    
    # Schoolbook for col 6
    U_25[0, 21] = 1; U_25[1, 22] = 1; U_25[2, 23] = 1; U_25[3, 24] = 1
    V_25[0*7 + 6, 21] = 1; V_25[1*7 + 6, 22] = 1; V_25[0*7 + 6, 23] = 1; V_25[1*7 + 6, 24] = 1
    W_25[6*2 + 0, 21] = 1; W_25[6*2 + 0, 22] = 1; W_25[6*2 + 0, 23] = 1; W_25[6*2 + 1, 24] = 1

    # Try to find Rank 24 using a global algebraic optimization
    # We look for a rank-3 decomposition for the last column (2x2 * 2x1)
    # A (2x2), B (2x1), C (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # We want 3 bilinear forms M1, M2, M3 such that C0, C1 are linear combinations of M1, M2, M3
    # and M's are products of linear forms in A and B.
    
    # Let's try to find a rank-3 solution for the last column
    # Variables:
    # U_last: 4x3 (coefficients for A00, A01, A10, A11)
    # V_last: 2x3 (coefficients for B0, B1)
    # W_last: 2x3 (coefficients for C0, C1)
    
    # Total variables: 4*3 + 2*3 + 2*3 = 12 + 6 + 6 = 24
    
    # Residuals:
    # For each (i, k) where i in {0,1} (row of A/C), k in {0,1} (col of B/C):
    # sum_r U[i, r] * V[j, r] * W[k, r] = delta_{i,k}
    # Wait, the tensor is T[i, j, k] = A_i * B_j -> C_k
    # Here A is 2x2, B is 2x1, C is 2x1.
    # Indices:
    # A: 0,1,2,3 (00, 01, 10, 11)
    # B: 0,1 (0, 1)
    # C: 0,1 (0, 1)
    # T[i, j, k] = 1 if A_i * B_j contributes to C_k
    # C0 = A00*B0 + A01*B1 -> T[0,0,0]=1, T[1,1,0]=1
    # C1 = A10*B0 + A11*B1 -> T[2,0,1]=1, T[3,1,1]=1
    
    # Residuals:
    # r0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + U[0,2]*V[0,2]*W[0,2] - 1 = 0
    # r1: U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] + U[1,2]*V[1,2]*W[0,2] - 1 = 0
    # r2: U[2,0]*V[0,0]*W[1,0] + U[2,1]*V[0,1]*W[1,1] + U[2,2]*V[0,2]*W[1,2] - 1 = 0
    # r3: U[3,0]*V[1,0]*W[1,0] + U[3,1]*V[1,1]*W[1,1] + U[3,2]*V[1,2]*W[1,2] - 1 = 0
    
    # And all other 4*2*2 - 4 = 12 entries must be 0.
    # Total residuals: 16 - 4 = 12 zeros + 4 ones = 16 constraints.
    # Variables: 24.
    # Underdetermined system, likely solvable.
    
    def last_col_residuals(x):
        U_l = x[:12].reshape(4, 3)
        V_l = x[12:18].reshape(2, 3)
        W_l = x[18:24].reshape(2, 3)
        
        res = []
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    s = np.dot(U_l[i, :], V_l[j, :] * W_l[k, :])
                    if (i == 0 and j == 0 and k == 0) or (i == 1 and j == 1 and k == 0) or \
                       (i == 2 and j == 0 and k == 1) or (i == 3 and j == 1 and k == 1):
                        res.append(s - 1.0)
                    else:
                        res.append(s)
        return np.array(res)
    
    best_res = float('inf')
    best_x = None
    
    if time.time() < deadline - 20:
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 12345)
            # Scale: values should be around 1
            x0 = np.random.randn(24) * 1.0
            
            try:
                sol = least_squares(last_col_residuals, x0, method='lm', 
                                   max_nfev=1000, xtol=1e-12, ftol=1e-12)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-12:
                        break
            except Exception:
                pass
        
        # If we found a good solution, polish it
        if best_x is not None and best_res < 1e-6:
            try:
                sol_polish = least_squares(last_col_residuals, best_x, method='lm',
                                          max_nfev=5000, xtol=1e-14, ftol=1e-14)
                if sol_polish.cost < best_res:
                    best_res = sol_polish.cost
                    best_x = sol_polish.x
            except Exception:
                pass
        
        # Check if we have a valid rank-3 solution for the last column
        if best_x is not None and best_res < 1e-10:
            U_l = best_x[:12].reshape(4, 3)
            V_l = best_x[12:18].reshape(2, 3)
            W_l = best_x[18:24].reshape(2, 3)
            
            # Try rounding to half-integers
            U_l_rounded = np.round(U_l * 2) / 2
            V_l_rounded = np.round(V_l * 2) / 2
            W_l_rounded = np.round(W_l * 2) / 2
            
            valid = True
            res_check = last_col_residuals(np.concatenate([U_l_rounded.flatten(), V_l_rounded.flatten(), W_l_rounded.flatten()]))
            if np.max(np.abs(res_check)) > 1e-6:
                # Try rounding to integers
                U_l_rounded = np.round(U_l)
                V_l_rounded = np.round(V_l)
                W_l_rounded = np.round(W_l)
                res_check = last_col_residuals(np.concatenate([U_l_rounded.flatten(), V_l_rounded.flatten(), W_l_rounded.flatten()]))
                if np.max(np.abs(res_check)) > 1e-6:
                    valid = False
            
            if valid:
                U_24 = np.zeros((4, 24))
                V_24 = np.zeros((14, 24))
                W_24 = np.zeros((14, 24))
                
                # First 21 columns are Strassen
                U_24[:, 0:7] += U_b1
                V_24[:, 0:7] += V_b1
                W_24[:, 0:7] += W_b1
                
                U_24[:, 7:14] += U_b2
                V_24[:, 7:14] += V_b2
                W_24[:, 7:14] += W_b2
                
                U_24[:, 14:21] += U_b3
                V_24[:, 14:21] += V_b3
                W_24[:, 14:21] += W_b3
                
                # Last 3 columns are the new rank-3 solution
                U_24[:, 21:24] += U_l_rounded
                # V_l is 2x3, corresponding to B[0,6] and B[1,6]
                # Global V indices: 0*7+6 = 6, 1*7+6 = 13
                V_24[6, 21:24] += V_l_rounded[0, :]
                V_24[13, 21:24] += V_l_rounded[1, :]
                # W_l is 2x3, corresponding to C[0,6] and C[1,6]
                # Global W indices: 6*2+0 = 12, 6*2+1 = 13
                W_24[12, 21:24] += W_l_rounded[0, :]
                W_24[13, 21:24] += W_l_rounded[1, :]
                
                return {"U": U_24.tolist(), "V": V_24.tolist(), "W": W_24.tolist()}

    return {"U": U_25.tolist(), "V": V_25.tolist(), "W": W_25.tolist()}
