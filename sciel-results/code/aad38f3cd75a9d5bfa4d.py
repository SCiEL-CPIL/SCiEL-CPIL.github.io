"""Matrix multiplication 2x5x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower to upper; schoolbook is n*m*p = 60
    for rank in range(20, 61):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, deadline):
    max_restarts = 50
    max_iter = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + restart * 7919)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # --- Update U rows ---
            # For each (i, j): solve M^T * u = 1_p where M[k, r] = V[jp+k, r] * W[kn+i, r]
            for i in range(n):
                for j in range(m):
                    # Build p x rank matrix: M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    M = V[j*p:(j+1)*p, :] * W[0:p*n:n, i:i]  # W[k*n+i] for k=0..p-1
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # --- Update V rows ---
            # For each (j, k): solve N^T * v = 1_n where N[i, r] = U[im+j, r] * W[kn+i, r]
            for j in range(m):
                for k in range(p):
                    # Build n x rank matrix: N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    N_mat = U[0:n*m:m, j:j] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # --- Update W rows ---
            # For each (k, i): solve P^T * w = 1_m where P[j, r] = U[im+j, r] * V[jp+k, r]
            for k in range(p):
                for i in range(n):
                    # Build m x rank matrix: P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    P_mat = U[i*m:(i+1)*m, :] * V[0:m*p:p, k:k]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence every 10 iterations
            if it % 10 == 9:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(val - 1))
                if max_err < 1e-7:
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
    
    return None

def _rationalize(U, V, W, n, m, p):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                if abs(val - 1) > 1e-9:
                    return None
    
    rank = U.shape[1]
    U_list = [[float(U_r[i,j]) for j in range(rank)] for i in range(U_r.shape[0])]
    V_list = [[float(V_r[i,j]) for j in range(rank)] for i in range(V_r.shape[0])]
    W_list = [[float(W_r[i,j]) for j in range(rank)] for i in range(W_r.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
