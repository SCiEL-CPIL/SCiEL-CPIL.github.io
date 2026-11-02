import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, r, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False, abs(val - 1.0)
        return True, 0.0
    
    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to reduce rank using iterative optimization
    if time.time() < deadline - 15:
        for target_rank in range(rank_best - 1, 40, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            max_restarts = 15 if target_rank >= 50 else 8
            
            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 7919 + target_rank * 13)
                
                # Initialize near rank-55 solution with random perturbations
                U_init = U_best[:, :target_rank].copy()
                V_init = V_best[:, :target_rank].copy()
                W_init = W_best[:, :target_rank].copy()
                
                # Add small random noise
                U_init += 0.01 * np.random.randn(nm, target_rank)
                V_init += 0.01 * np.random.randn(mp, target_rank)
                W_init += 0.01 * np.random.randn(pn, target_rank)
                
                # ALS-style optimization
                U_cur, V_cur, W_cur = U_init, V_init, W_init
                best_err = float('inf')
                
                for iteration in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U
                    for idx in range(nm):
                        i, j = divmod(idx, m)
                        # U[idx, :] should satisfy: for all k, sum_r U[idx,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                        # Fix V, W, solve for U[idx, :]
                        # This is a linear system if we fix V and W
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        b_vec = np.ones(p)
                        # Least squares: min ||A_mat @ u - b_vec||
                        u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_cur[idx, :] = u_sol
                    
                    # Update V
                    for idx in range(mp):
                        j, k = divmod(idx, p)
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        b_vec = np.ones(n)
                        v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_cur[idx, :] = v_sol
                    
                    # Update W
                    for idx in range(pn):
                        k, i = divmod(idx, n)
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b_vec = np.ones(m)
                        w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_cur[idx, :] = w_sol
                    
                    # Check error
                    err_val = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_cur[i*m + j, :], V_cur[j*p + k, :] * W_cur[k*n + i, :])
                                err_val = max(err_val, abs(val - 1.0))
                    
                    if err_val < best_err:
                        best_err = err_val
                        if err_val < 1e-7:
                            found = True
                            break
                    
                    if err_val < 1e-9:
                        found = True
                        break
                
                if found:
                    U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                    break
    
    # Convert to lists for output
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
