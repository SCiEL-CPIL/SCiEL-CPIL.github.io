import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Fallback: Rank 50 decomposition from current program
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution
    def try_rank(r):
        if time.time() > deadline - 10:
            return None
            
        # We will fix W to be sparse (e.g., 2 non-zeros per column) and solve for U, V
        # Actually, let's fix W to be a random sparse matrix with small integers
        # And solve for U given V? Or iterate?
        
        # Strategy: 
        # 1. Random sparse V (27 x r) and W (18 x r)
        # 2. For each i, j, solve M[i,j] @ U[i,j] = 1
        # 3. Check validity
        
        for _ in range(20):  # 20 trials per rank
            if time.time() > deadline - 5:
                return None
                
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Fill V and W with random sparse integers
            # Ensure some non-zeros
            for c in range(r):
                # V column
                indices = np.random.choice(m_rows, size=max(1, r//10), replace=False)
                for idx in indices:
                    V[idx, c] = np.random.choice([-1, 1, 2, -2])
                # W column
                indices = np.random.choice(p_rows, size=max(1, r//10), replace=False)
                for idx in indices:
                    W[idx, c] = np.random.choice([-1, 1, 2, -2])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build M (p x r)
                    M = np.zeros((p, r), dtype=np.int64)
                    for k in range(p):
                        # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M @ u = 1
                    # Least squares or exact?
                    # If M has rank < p, no solution or infinite.
                    # Use lstsq
                    u, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    
                    # Check if solution is integer-like
                    u_int = np.round(u).astype(np.int64)
                    
                    # Verify exactness
                    if not np.allclose(M @ u_int, np.ones(p)):
                        valid = False
                        break
                        
                    U[i*m+j, :] = u_int
                    
            if not valid:
                continue
                
            # Full verification
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Should be 1 for all i,j,k
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            success = False
                            break
                    if not success: break
                if not success: break
                
            if success:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        return None

    # Try ranks from 49 down to 36
    for r in range(49, 35, -1):
        res = try_rank(r)
        if res:
            return res
            
    # Fallback to rank 50
    return current_solution()
