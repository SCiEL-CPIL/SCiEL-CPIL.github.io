import time
import numpy as np
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    def check_solution(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = len(U[0])
        U_np = np.array(U, dtype=float).reshape(n, m, R)
        V_np = np.array(V, dtype=float).reshape(m, p, R)
        W_np = np.array(W, dtype=float).reshape(p, n, R)
        
        # Compute M[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Using einsum: 'ijr,jkr,kir -> ijk'
        try:
            M = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np)
        except:
            return False
            
        return np.allclose(M, 1.0, atol=1e-9)

    def solve_block(A, b):
        # Solve A x = b using least squares
        # A: (num_eq, R), b: (num_eq,)
        # Returns x: (R,)
        if A.size == 0:
            return np.zeros(A.shape[1])
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            return x
        except:
            return np.zeros(A.shape[1])

    # Try to find a better rank
    # Known lower bound for M_{2,3,4} is likely around 20.
    # We try ranks from 20 to 23.
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
            
        # Multiple random restarts for each rank
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small integers
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)], dtype=float)
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)], dtype=float)
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)], dtype=float)
            
            # Reshape for easy indexing
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            converged = False
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A[k, r] = V[j, k, r] * W[k, i, r]
                        A_u = np.zeros((p, target_rank))
                        for k in range(p):
                            A_u[k, :] = V_r[j, k, :] * W_r[k, i, :]
                        b_u = np.ones(p)
                        U_r[i, j, :] = solve_block(A_u, b_u)
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # A[i, r] = U[i, j, r] * W[k, i, r]
                        A_v = np.zeros((n, target_rank))
                        for i in range(n):
                            A_v[i, :] = U_r[i, j, :] * W_r[k, i, :]
                        b_v = np.ones(n)
                        V_r[j, k, :] = solve_block(A_v, b_v)
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        # A[j, r] = U[i, j, r] * V[j, k, r]
                        A_w = np.zeros((m, target_rank))
                        for j in range(m):
                            A_w[j, :] = U_r[i, j, :] * V_r[j, k, :]
                        b_w = np.ones(m)
                        W_r[k, i, :] = solve_block(A_w, b_w)
                        
                # Check convergence / integrality
                # Check if values are close to half-integers
                U_check = np.round(U_r * 2) / 2
                V_check = np.round(V_r * 2) / 2
                W_check = np.round(W_r * 2) / 2
                
                if np.allclose(U_r, U_check, atol=1e-6) and np.allclose(V_r, V_check, atol=1e-6) and np.allclose(W_r, W_check, atol=1e-6):
                    # Verify exactness
                    U_int = U_check.reshape(n * m, target_rank).tolist()
                    V_int = V_check.reshape(m * p, target_rank).tolist()
                    W_int = W_check.reshape(p * n, target_rank).tolist()
                    
                    if check_solution(U_int, V_int, W_int):
                        return {"U": U_int, "V": V_int, "W": W_int}
                    
                    # If it converged to half-integers but didn't pass check, 
                    # it might be a local minimum that doesn't satisfy the equations exactly 
                    # or the check failed due to floating point issues in einsum?
                    # Let's try to polish with exact rational arithmetic? 
                    # Or just accept if close enough? The task requires EXACT.
                    # If check_solution fails, we continue searching.
                    # But we might be stuck. Let's perturb slightly.
                    U_r += np.random.normal(0, 0.1, U_r.shape)
                    V_r += np.random.normal(0, 0.1, V_r.shape)
                    W_r += np.random.normal(0, 0.1, W_r.shape)

    # If no better solution found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
