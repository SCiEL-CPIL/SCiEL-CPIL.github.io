import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def get_strassen_block(k_start, col_start):
        """
        Returns U, V, W contributions for a 2x2x2 Strassen block.
        """
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
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * k_start)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * k_start + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * k_start)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * k_start + 1)
                V_block[v_idx, r] = coeff
            
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * k_start) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * k_start + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * k_start) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * k_start + 1) * n + 1
                W_block[w_idx, r] = coeff
        
        return U_block, V_block, W_block

    def find_2x2x3_rank10():
        """
        Attempts to find a Rank 10 decomposition for 2x2x3 matrix multiplication.
        """
        if time.time() > deadline - 20:
            return None
            
        R = 10
        # Variables: U (4x10), V (6x10), W (6x10)
        
        def residuals(x):
            U = x[:4*R].reshape(4, R)
            V = x[4*R:4*R+6*R].reshape(6, R)
            W = x[4*R+6*R:].reshape(6, R)
            
            res = np.zeros(12)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0.0
                        for r in range(R):
                            s += U[i, r] * V[j*3+k, r] * W[k*2+i, r]
                        res[idx] = s - 1.0
                        idx += 1
            return res

        best_res = float('inf')
        best_x = None
        
        # Try more random initializations
        num_restarts = 50
        for attempt in range(num_restarts):
            if time.time() > deadline - 20:
                break
            np.random.seed(attempt * 12345)
            # Scale initial values to help convergence
            x0 = np.random.randn(16*R) * 1.0
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_x is not None and best_res < 1e-6:
            U = best_x[:4*R].reshape(4, R)
            V = best_x[4*R:4*R+6*R].reshape(6, R)
            W = best_x[4*R+6*R:].reshape(6, R)
            
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0.0
                        for r in range(R):
                            s += U_rounded[i, r] * V_rounded[j*3+k, r] * W_rounded[k*2+i, r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_rounded, V_rounded, W_rounded
        
        return None

    # --- Attempt Rank 24 ---
    U_24 = np.zeros((4, 24))
    V_24 = np.zeros((14, 24))
    W_24 = np.zeros((14, 24))
    
    # Block 1: Cols 0,1 (Strassen)
    U_b1, V_b1, W_b1 = get_strassen_block(0, 0)
    U_24[:, 0:7] += U_b1
    V_24[:, 0:7] += V_b1
    W_24[:, 0:7] += W_b1
    
    # Block 2: Cols 2,3 (Strassen)
    U_b2, V_b2, W_b2 = get_strassen_block(1, 7)
    U_24[:, 7:14] += U_b2
    V_24[:, 7:14] += V_b2
    W_24[:, 7:14] += W_b2
    
    # Block 3: Cols 4,5,6 (2x2x3 Rank 10)
    block3 = find_2x2x3_rank10()
    
    if block3 is not None:
        U_b3, V_b3, W_b3 = block3
        U_24[:, 14:24] += U_b3
        for j in range(2):
            for k_loc in range(3):
                k_glob = 4 + k_loc
                V_24[j*7 + k_glob, 14:24] += V_b3[j*3 + k_loc, :]
        
        for k_loc in range(3):
            for i in range(2):
                k_glob = 4 + k_loc
                W_24[k_glob*2 + i, 14:24] += W_b3[k_loc*2 + i, :]
        
        # Final verification
        if time.time() < deadline - 5:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.dot(U_24[a_idx, :], V_24[b_idx, :] * W_24[c_idx, :])
                        if abs(val - 1.0) > max_err:
                            max_err = abs(val - 1.0)
            
            if max_err < 1e-6:
                return {"U": U_24.tolist(), "V": V_24.tolist(), "W": W_24.tolist()}
    
    # --- Fallback to Rank 25 ---
    U_25 = np.zeros((4, 25))
    V_25 = np.zeros((14, 25))
    W_25 = np.zeros((14, 25))
    
    U_b1, V_b1, W_b1 = get_strassen_block(0, 0)
    U_25[:, 0:7] += U_b1
    V_25[:, 0:7] += V_b1
    W_25[:, 0:7] += W_b1
    
    U_b2, V_b2, W_b2 = get_strassen_block(1, 7)
    U_25[:, 7:14] += U_b2
    V_25[:, 7:14] += V_b2
    W_25[:, 7:14] += W_b2
    
    U_b3, V_b3, W_b3 = get_strassen_block(2, 14)
    U_25[:, 14:21] += U_b3
    V_25[:, 14:21] += V_b3
    W_25[:, 14:21] += W_b3
    
    # Block 4: Col 6 (Schoolbook)
    U_25[0, 21] = 1
    U_25[1, 22] = 1
    U_25[2, 23] = 1
    U_25[3, 24] = 1
    
    V_25[0*7 + 6, 21] = 1
    V_25[1*7 + 6, 22] = 1
    V_25[0*7 + 6, 23] = 1
    V_25[1*7 + 6, 24] = 1
    
    W_25[6*2 + 0, 21] = 1
    W_25[6*2 + 0, 22] = 1
    W_25[6*2 + 1, 23] = 1
    W_25[6*2 + 1, 24] = 1
    
    return {"U": U_25.tolist(), "V": V_25.tolist(), "W": W_25.tolist()}
