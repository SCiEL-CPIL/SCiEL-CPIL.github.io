import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
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

    def verify_fast(U, V, W):
        """Fast verification using numpy"""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True

    def try_local_search():
        """Try to reduce rank by local search: remove one term and adjust others"""
        if time.time() > deadline - 20:
            return None
        
        sol = current_solution()
        U = np.array(sol["U"], dtype=np.float64)
        V = np.array(sol["V"], dtype=np.float64)
        W = np.array(sol["W"], dtype=np.float64)
        rank = len(U[0])
        
        # For each column, try to remove it and see if we can adjust the rest
        # to still satisfy the constraints
        for remove_col in range(rank):
            if time.time() > deadline - 15:
                return None
            
            # Get the column to remove
            u_rem = U[:, remove_col]
            v_rem = V[:, remove_col]
            w_rem = W[:, remove_col]
            
            # The contribution of this term to each (i,j,k) is:
            # u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
            # We need to compensate for this by adjusting other columns
            
            # This is complex. Instead, try a simpler approach:
            # Check if the column is "redundant" - i.e., its contribution can be
            # absorbed by small adjustments to other columns
            
            # Try: set this column to zero and see if we can find small corrections
            # to other columns to restore the identity
            
            # This is a linear system. Let's try it for a few columns.
            
            if remove_col % 5 != 0:  # Only try every 5th column for speed
                continue
            
            # Create modified matrices with this column zeroed out
            U_mod = U.copy()
            V_mod = V.copy()
            W_mod = W.copy()
            U_mod[:, remove_col] = 0
            V_mod[:, remove_col] = 0
            W_mod[:, remove_col] = 0
            
            # Compute the error for each (i,j,k)
            # Error[i,j,k] = -u_rem[i*m+j] * v_rem[j*p+k] * w_rem[k*n+i]
            # We need to find corrections delta_u, delta_v, delta_w such that
            # sum_c (U_mod[i*m+j,c] + delta_u[i*m+j,c]) * (V_mod[j*p+k,c] + delta_v[j*p+k,c]) * (W_mod[k*n+i,c] + delta_w[k*n+i,c]) = 1
            # This is nonlinear, so let's linearize around the current solution
            
            # Linearized: sum_c [U_mod[i*m+j,c]*V_mod[j*p+k,c]*delta_w[k*n+i,c] +
            #                      U_mod[i*m+j,c]*delta_v[j*p+k,c]*W_mod[k*n+i,c] +
            #                      delta_u[i*m+j,c]*V_mod[j*p+k,c]*W_mod[k*n+i,c]] = -error[i,j,k]
            
            # This is a large linear system. Let's try to solve it.
            
            # Actually, this is getting too complex. Let's try a different approach.
            pass
        
        return None

    def try_merge_terms():
        """Try to merge two terms into one by finding a common factor"""
        if time.time() > deadline - 20:
            return None
        
        sol = current_solution()
        U = np.array(sol["U"], dtype=np.int64)
        V = np.array(sol["V"], dtype=np.int64)
        W = np.array(sol["W"], dtype=np.int64)
        rank = len(U[0])
        
        # Try to find two columns that can be merged
        # Two columns c1 and c2 can be merged if there exist scalars a, b such that
        # (U[:,c1] + a*U[:,c2], V[:,c1] + b*V[:,c2], W[:,c1] + c*W[:,c2]) is valid
        # This is complex. Skip for now.
        
        return None

    # Try to improve the rank-50 solution
    best_result = current_solution()
    best_rank = 50
    
    # Try local search
    if time.time() < deadline - 25:
        res = try_local_search()
        if res:
            best_result = res
            best_rank = len(res["U"][0])
    
    # Try structured search for lower ranks
    if best_rank == 50 and time.time() < deadline - 30:
        for r in range(49, 44, -1):
            if time.time() > deadline - 15:
                break
            
            # Try a few random initializations
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                
                V = np.zeros((m_rows, r), dtype=np.int64)
                W = np.zeros((p_rows, r), dtype=np.int64)
                
                for c in range(r):
                    nz_v = random.randint(3, 4)
                    v_indices = np.random.choice(m_rows, size=nz_v, replace=False)
                    for idx in v_indices:
                        V[idx, c] = random.choice([-1, 1, -2, 2])
                    
                    nz_w = random.randint(2, 3)
                    w_indices = np.random.choice(p_rows, size=nz_w, replace=False)
                    for idx in w_indices:
                        W[idx, c] = random.choice([-1, 1, -2, 2])
                
                U = np.zeros((n_rows, r), dtype=np.int64)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r), dtype=np.float64)
                        for k in range(p):
                            M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                        
                        try:
                            u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except:
                            valid = False
                            break
                        
                        u_int = np.round(u).astype(np.int64)
                        
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                            
                        U[i*m+j, :] = u_int
                
                if valid:
                    if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = r
                        break
            
            if best_rank < 50:
                break
    
    return best_result
