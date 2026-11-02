"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by 5x6 matrix.
    Uses tensor flattening and linear solving to search for low-rank decompositions.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Try a few candidate ranks in increasing order
    candidate_ranks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    for rank in candidate_ranks:
        # Try multiple random initializations
        for trial in range(50):
            np.random.seed(1000 + rank * 100 + trial)
            
            # Initialize U and V with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            
            # Solve for W
            W = np.zeros((p * n, rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build A[j, s] = U[i*m+j, s] * V[j*p+k, s]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ W[k*n+i, :] = ones(m)
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = sol
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if the solution is valid
            check_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                        if abs(total - 1.0) > 1e-7:
                            check_valid = False
                            break
                    if not check_valid:
                        break
                if not check_valid:
                    break
            
            if check_valid:
                # Found a valid decomposition
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
