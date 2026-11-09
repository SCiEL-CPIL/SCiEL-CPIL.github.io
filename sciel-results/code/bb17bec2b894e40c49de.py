import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Trivial decomposition rank = n*m*p = 50
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    # Precompute indices for faster evaluation
    idx_U = [(i * m + j) for i in range(n) for j in range(m)]
    idx_V = [(j * p + k) for j in range(m) for k in range(p)]
    idx_W = [(k * n + i) for k in range(p) for i in range(n)]
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Start from rank 49 down to 30
    for rank in range(rank_trivial - 1, max(25, rank_trivial - 20), -1):
        if time.time() > deadline:
            break
        
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + attempt)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Try a few iterations of alternating minimization
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1 for each k
                # So: U[i*m+j, :] is a solution to a linear system
                for i in range(n):
                    for j in range(m):
                        # Build the linear system: A @ x = b
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # b[k] = 1
                        A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        b = np.ones(p)
                        # Solve A @ x = b in least squares sense
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # b[i] = 1
                        A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        b = np.ones(n)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # b[j] = 1
                        A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        b = np.ones(m)
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
            
            # Check if we found a valid decomposition
            if check_decomposition(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Found valid decomposition for this rank
    
    if best_rank < rank_trivial:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
