"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill
    
    # Schoolbook solution (rank = n*m*p = 84)
    r_sb = n * m * p
    U_sb = np.zeros((n*m, r_sb))
    V_sb = np.zeros((m*p, r_sb))
    W_sb = np.zeros((p*n, r_sb))
    for col in range(r_sb):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        U_sb[i*m+j, col] = 1.0
        V_sb[j*p+k, col] = 1.0
        W_sb[k*n+i, col] = 1.0
    
    def check_solution(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        return np.allclose(arr * 2, np.round(arr * 2), atol=tol)
    
    def als_decompose(rank, max_iter=20, seed=0):
        """Efficient ALS for tensor decomposition."""
        rng = np.random.RandomState(seed)
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = rng.choice(vals, size=(n*m, rank))
        V = rng.choice(vals, size=(m*p, rank))
        W = rng.choice(vals, size=(p*n, rank))
        
        for c in range(rank):
            if np.allclose(U[:, c], 0):
                U[rng.randint(n*m), c] = 1.0
            if np.allclose(V[:, c], 0):
                V[rng.randint(m*p), c] = 1.0
            if np.allclose(W[:, c], 0):
                W[rng.randint(p*n), c] = 1.0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new[k*n+i, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            if check_solution(U, V, W, tol=1e-6):
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if check_solution(U, V, W, tol=1e-8) and is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    return U, V, W
        
        if check_solution(U, V, W, tol=1e-8):
            return U, V, W
        return None
    
    # Try to find rank 83 (one less than schoolbook)
    best_rank = r_sb
    best_solution = (U_sb, V_sb, W_sb)
    
    start_time = time.time()
    # Try rank 83 with a few restarts, limited time
    max_restarts_83 = 3
    for restart in range(max_restarts_83):
        if time.time() > deadline:
            break
        result = als_decompose(83, max_iter=15, seed=100+restart)
        if result is not None:
            best_rank = 83
            best_solution = result
            break
    
    # If we still have time, try rank 82
    if best_rank == 83 and time.time() < deadline - 30:
        result = als_decompose(82, max_iter=10, seed=200)
        if result is not None:
            best_rank = 82
            best_solution = result
    
    U, V, W = best_solution
    
    # Final safety check
    if not check_solution(U, V, W, tol=1e-6):
        U, V, W = U_sb, V_sb, W_sb
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
