import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    # Try ranks from 84 down to 50
    for rank in range(84, 49, -1):
        # Try to find a decomposition of this rank
        # Fix V and W randomly, solve for U
        for attempt in range(100):
            np.random.seed(attempt)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            U = np.zeros((n * m, rank))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Solve for U[i*m+j, :]
                    # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is a linear system: M @ U_row = ones
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    
                    # Solve M @ U_row = ones(p)
                    try:
                        U_row, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        # Check if the solution is valid
                        if np.allclose(M @ U_row, np.ones(p), atol=1e-10):
                            U[i * m + j, :] = U_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                T_reconstructed = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_reconstructed[i, j, k] = np.sum(
                                U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                            )
                
                if np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
                    # Found a valid decomposition
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback to naive rank 84
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
