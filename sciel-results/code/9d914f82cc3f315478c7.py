"""Matrix multiplication 2,3,7 rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    # Schoolbook decomposition (rank 42) as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(computed - 1.0) > 1e-8:
                        return False
        return True
    
    best_rank = rank_school
    best_solution = (U_school, V_school, W_school)
    
    # Try to find lower rank solutions using alternating least squares
    # with random restarts
    
    target_ranks = [35, 32, 30, 28, 26, 25, 24, 23, 22, 21]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for this rank
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Alternating least squares
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Update W: for each (i,j,k), W[k*n+i, :] = a / ||a||^2
                # where a_r = U[i*m+j, r] * V[j*p+k, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = U[i * m + j, :] * V[j * p + k, :]
                            norm_sq = np.dot(a, a)
                            if norm_sq > 1e-10:
                                W[k * n + i, :] = a / norm_sq
                
                # Update U: for each (i,j), U[i*m+j, :] = b / ||b||^2
                # where b_r = V[j*p+k, r] * W[k*n+i, r] summed over k... 
                # Actually, we need sum_k sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for each k
                # This is more complex. Let's do it differently.
                # For fixed (i,j), we need: for all k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # Let c_k,r = V[j*p+k,r] * W[k*n+i,r]
                # Then sum_r U[i*m+j,r] * c_k,r = 1 for all k
                # This is a system of p equations in target_rank unknowns
                # We can solve it using least squares
                for i in range(n):
                    for j in range(m):
                        # Build the system: C[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        C = np.zeros((p, target_rank))
                        for k in range(p):
                            C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve: U[i*m+j, :] @ C.T = ones(p)
                        # i.e., C @ U[i*m+j, :].T = ones(p)
                        # Use least squares
                        try:
                            U[i * m + j, :], residuals, rank_u, sv = np.linalg.lstsq(C, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V: similar
                for j in range(m):
                    for k in range(p):
                        # Build: D[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        D = np.zeros((n, target_rank))
                        for i in range(n):
                            D[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        # Solve: V[j*p+k, :] @ D.T = ones(n)
                        try:
                            V[j * p + k, :], residuals, rank_v, sv = np.linalg.lstsq(D, np.ones(n), rcond=None)
                        except:
                            pass
            
            if verify_solution(U, V, W, target_rank):
                best_rank = target_rank
                best_solution = (U, V, W)
                break
        
        if best_solution is not (U_school, V_school, W_school):
            break
    
    U_sol, V_sol, W_sol = best_solution
    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

