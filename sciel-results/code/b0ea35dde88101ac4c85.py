import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    
    # Trivial rank-42 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(N_U)]
    V_trivial = [[0] * rank_trivial for _ in range(N_V)]
    W_trivial = [[0] * rank_trivial for _ in range(N_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][column] = 1
                V_trivial[j * p + k][column] = 1
                W_trivial[k * n + i][column] = 1
                column += 1
                
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = rank_trivial
    
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Build the target tensor T[i,j,k] = 1 if (i,j,k) is a valid index
    # T has shape (n, m, p) and T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed as W[k*n+i, r] per the trace convention
    
    # For ALS, we flatten the tensor and use unfolding
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try ranks from a reasonable upper bound down to the lower bound
    # Lower bound: max(n*m, m*p, n*p) = max(6, 21, 14) = 21
    
    np.random.seed(42)
    
    # Try various ranks
    for rank in range(35, 20, -1):
        if time.time() > deadline:
            break
            
        for trial in range(5):
            if time.time() > deadline:
                break
                
            # Initialize randomly
            U = np.random.randn(N_U, rank) * 0.1
            V = np.random.randn(N_V, rank) * 0.1
            W = np.random.randn(N_W, rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j), we have: T[i,j,:] = U[i*m+j, :] * diag(V[j*p+k,:] * W[k*n+i,:] for all k)
                # More precisely: for each k, T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # So: T[i,j,:] (length p) = U[i*m+j,:] (length rank) @ M_{ij}^T
                # where M_{ij}[k,r] = V[j*p+k,r] * W[k*n+i,r]
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build M_{ij}: shape (p, rank)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve: T[i,j,:] = U[i*m+j,:] @ M.T
                        # U[i*m+j,:] = T[i,j,:] @ pinv(M.T)
                        # Or: (M @ M.T) @ U[i*m+j,:]^T = M @ T[i,j,:]^T
                        A = M @ M.T  # (p, p)
                        b = M @ np.ones(p)  # T[i,j,k] = 1 for all k
                        try:
                            U[i*m+j, :] = np.linalg.solve(A + 1e-10*np.eye(p), b)
                        except:
                            U[i*m+j, :] = np.linalg.lstsq(A + 1e-10*np.eye(p), b, rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ N_{jk}.T
                        # where N_{jk}[i,r] = U[i*m+j,r] * W[k*n+i,r]
                        N = np.zeros((n, rank))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        A = N @ N.T  # (n, n)
                        b = N @ np.ones(n)  # T[i,j,k] = 1 for all i
                        try:
                            V[j*p+k, :] = np.linalg.solve(A + 1e-10*np.eye(n), b)
                        except:
                            V[j*p+k, :] = np.linalg.lstsq(A + 1e-10*np.eye(n), b, rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ P_{ki}.T
                        # where P_{ki}[j,r] = U[i*m+j,r] * V[j*p+k,r]
                        P = np.zeros((m, rank))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        A = P @ P.T  # (m, m)
                        b = P @ np.ones(m)  # T[i,j,k] = 1 for all j
                        try:
                            W[k*n+i, :] = np.linalg.solve(A + 1e-10*np.eye(m), b)
                        except:
                            W[k*n+i, :] = np.linalg.lstsq(A + 1e-10*np.eye(m), b, rcond=None)[0]
                
                # Check convergence: compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                
                if error < 1e-12:
                    converged = True
                    break
            else:
                if time.time() > deadline:
                    break
            
            if converged:
                # Check if this rank is better
                if rank < best_rank:
                    # Convert to list of lists with rounded values
                    U_list = [[round(x, 10) if abs(x) > 1e-9 else 0 for x in row] for row in U]
                    V_list = [[round(x, 10) if abs(x) > 1e-9 else 0 for x in row] for row in V]
                    W_list = [[round(x, 10) if abs(x) > 1e-9 else 0 for x in row] for row in W]
                    
                    # Verify exactness
                    verify_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] for r in range(rank))
                                if abs(val - 1.0) > 1e-8:
                                    verify_ok = False
                                    break
                            if not verify_ok:
                                break
                        if not verify_ok:
                            break
                    
                    if verify_ok:
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        # Try to go lower
                        break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
