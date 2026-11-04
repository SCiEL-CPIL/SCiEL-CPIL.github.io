import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # We want to find U, V, W of rank R
    # U: (N_U, R), V: (N_V, R), W: (N_W, R)
    
    # Start with a reasonable rank
    # Schoolbook is 45.
    # Let's try to find a lower rank.
    # We will try ranks from 20 up to 45.
    # If we find a valid decomposition, return it.
    
    # To save time, we will try a few ranks.
    # The lower bound is 15.
    # Let's try 20, 25, 30, 35, 40, 45.
    
    # We will use ALS.
    
    def als_step(U, V, W, T, factor='U'):
        if factor == 'U':
            # Solve for U
            # For each u, solve sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w]
            # This is a linear system in U[u, :]
            # Matrix A has shape (N_V * N_W, R)
            # A[v*N_W + w, r] = V[v, r] * W[w, r]
            # b[v*N_W + w] = T[u, v, w]
            
            # Precompute A for all u? A is independent of u.
            # A is (N_V * N_W, R)
            A = np.einsum('vr,wr->vwr', V, W).reshape(N_V * N_W, -1)
            
            U_new = np.zeros_like(U)
            for u in range(N_U):
                b = T[u, :, :].reshape(-1)
                # Solve A x = b in least squares
                # If R is small, this is fast.
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    U_new[u, :] = sol
                except:
                    U_new[u, :] = U[u, :]
            return U_new
            
        elif factor == 'V':
            # Solve for V
            # For each v, solve sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w]
            # Matrix A has shape (N_U * N_W, R)
            # A[u*N_W + w, r] = U[u, r] * W[w, r]
            
            A = np.einsum('ur,wr->uwr', U, W).reshape(N_U * N_W, -1)
            
            V_new = np.zeros_like(V)
            for v in range(N_V):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    V_new[v, :] = sol
                except:
                    V_new[v, :] = V[v, :]
            return V_new
            
        elif factor == 'W':
            # Solve for W
            # For each w, solve sum_r U[u, r] V[v, r] W[w, r] = T[u, v, w]
            # Matrix A has shape (N_U * N_V, R)
            # A[u*N_V + v, r] = U[u, r] * V[v, r]
            
            A = np.einsum('ur,vr->uvr', U, V).reshape(N_U * N_V, -1)
            
            W_new = np.zeros_like(W)
            for w in range(N_W):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    W_new[w, :] = sol
                except:
                    W_new[w, :] = W[w, :]
            return W_new

    def check_error(U, V, W, T):
        # Compute T_approx = sum_r U[:, r] V[:, r] W[:, r]
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T_approx - T)

    # Try different ranks
    # We will use a deadline guard
    start_time = time.time()
    deadline = 120.0
    
    best_rank = 45
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook as fallback
    U_school = np.zeros((N_U, 45))
    V_school = np.zeros((N_V, 45))
    W_school = np.zeros((N_W, 45))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = 45
    
    # Try to find lower rank
    # We will try ranks 20, 25, 30, 35, 40
    # If time permits.
    
    ranks_to_try = [20, 25, 30, 35, 40]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + R)
            U = np.random.randn(N_U, R)
            V = np.random.randn(N_V, R)
            W = np.random.randn(N_W, R)
            
            # Run ALS
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                U = als_step(U, V, W, T, 'U')
                V = als_step(U, V, W, T, 'V')
                W = als_step(U, V, W, T, 'W')
                
                err = check_error(U, V, W, T)
                if err < 1e-6:
                    # Found a valid decomposition
                    # Convert to list of lists
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank == R:
                break
                
    # If we found a better rank, use it
    if best_rank < 45:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
    return {"U": U_out, "V": V_out, "W": W_out}
