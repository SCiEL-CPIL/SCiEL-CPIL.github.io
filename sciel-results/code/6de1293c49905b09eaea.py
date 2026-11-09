import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Construct the full tensor T[i, j, k] = 1 if i, j, k are valid indices
    # T is of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Try to find a lower rank decomposition using ALS
    # Lower bound for rank is roughly (n*m*p) / max(n,m,p) but let's just try a range
    # Known lower bound for matrix multiplication tensor rank is not trivially small.
    # Let's try ranks from 15 to 29.
    
    best_solution = None
    best_rank = rank_school
    
    # We will use ALS with random initialization
    rng = np.random.default_rng(42)
    
    # Pre-compute indices for faster access
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    for r in range(15, rank_school):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # We want to solve for w = W[k*n+i, :]
                    # T[i, :, k] (length m) = M @ w where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    U_i = U[i*m:(i+1)*m, :] # (m, r)
                    V_k = V[k::p, :] # (m, r) -- V rows are j*p+k, so for fixed k, rows k, k+p, k+2p...
                    # Wait, V has shape (m*p, r). Row index is j*p + k.
                    # For fixed k, the rows are k, p+k, 2p+k...
                    # So V_k = V[k::p, :]
                    
                    M = U_i * V_k # (m, r)
                    b = T[i, :, k] # (m,)
                    
                    # Solve M @ w = b
                    try:
                        w = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        w = np.zeros(r)
                    W_new[k*n+i, :] = w
            
            W = W_new
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # T[i, j, :] (length p) = M @ u where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    V_j = V[j*p:(j+1)*p, :] # (p, r)
                    W_i = W[i::n, :] # (p, r) -- W rows are k*n+i, so for fixed i, rows i, n+i, 2n+i...
                    
                    M = V_j * W_i # (p, r)
                    b = T[i, j, :] # (p,)
                    
                    try:
                        u = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        u = np.zeros(r)
                    U_new[i*m+j, :] = u
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # T[:, j, k] (length n) = M @ v where M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    U_j = U[j::m, :] # (n, r) -- U rows are i*m+j, so for fixed j, rows j, m+j, 2m+j...
                    W_k = W[k*n:(k+1)*n, :] # (n, r)
                    
                    M = U_j * W_k # (n, r)
                    b = T[:, j, k] # (n,)
                    
                    try:
                        v = np.linalg.lstsq(M, b, rcond=None)[0]
                    except:
                        v = np.zeros(r)
                    V_new[j*p+k, :] = v
            V = V_new
            
            # Check convergence / exactness
            # Reconstruct T_hat
            T_hat = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                        T_hat[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        
            error = np.linalg.norm(T - T_hat)
            if error < 1e-9:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_hat_round = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_hat_round[i, j, k] = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
                            
                if np.linalg.norm(T - T_hat_round) < 1e-9:
                    # Found exact half-integer solution
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                else:
                    # Return the floating point solution if it's exact enough? 
                    # The task says "integer/half-integer entries". 
                    # So we must return half-integers.
                    pass
                    
            if time.time() > deadline:
                break

    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
