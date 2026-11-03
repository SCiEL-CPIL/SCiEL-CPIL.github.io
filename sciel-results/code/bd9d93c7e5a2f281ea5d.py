import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try to find a lower rank decomposition
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank using ALS
    # Start from a rank slightly below schoolbook
    # Or start from random and optimize
    # Given the time limit, we try a few ranks
    
    for rank in range(lower_bound, schoolbook_rank):
        # Try to find a decomposition of this rank
        # Use ALS
        U_rand = np.random.randn(n * m, rank) * 0.1
        V_rand = np.random.randn(m * p, rank) * 0.1
        W_rand = np.random.randn(p * n, rank) * 0.1
        
        # Optimize
        for _ in range(100):
            # Update U
            for i in range(n):
                for j in range(m):
                    # Solve for U[i*m+j, :]
                    # Minimize sum_k (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
                    # This is a least squares problem
                    # Let x = U[i*m+j, :]
                    # A = sum_k outer(V[j*p+k, :] * W[k*n+i, :], V[j*p+k, :] * W[k*n+i, :])
                    # b = sum_k V[j*p+k, :] * W[k*n+i, :]
                    A = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for k in range(p):
                        v_vec = V_rand[j * p + k, :]
                        w_vec = W_rand[k * n + i, :]
                        A += np.outer(v_vec * w_vec, v_vec * w_vec)
                        b += v_vec * w_vec
                    A += 1e-10 * np.eye(rank)
                    try:
                        U_rand[i * m + j, :] = np.linalg.solve(A, b)
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for i in range(n):
                        u_vec = U_rand[i * m + j, :]
                        w_vec = W_rand[k * n + i, :]
                        A += np.outer(u_vec * w_vec, u_vec * w_vec)
                        b += u_vec * w_vec
                    A += 1e-10 * np.eye(rank)
                    try:
                        V_rand[j * p + k, :] = np.linalg.solve(A, b)
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, rank))
                    b = np.zeros(rank)
                    for j in range(m):
                        u_vec = U_rand[i * m + j, :]
                        v_vec = V_rand[j * p + k, :]
                        A += np.outer(u_vec * v_vec, u_vec * v_vec)
                        b += u_vec * v_vec
                    A += 1e-10 * np.eye(rank)
                    try:
                        W_rand[k * n + i, :] = np.linalg.solve(A, b)
                    except:
                        pass
        
        # Check error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_rand[i * m + j, c] * V_rand[j * p + k, c] * W_rand[k * n + i, c]
                    error += (val - 1.0) ** 2
        
        if error < 1e-8:
            best_rank = rank
            best_U = U_rand.copy()
            best_V = V_rand.copy()
            best_W = W_rand.copy()
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
