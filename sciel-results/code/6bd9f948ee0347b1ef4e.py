"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook rank = n*m*p = 75
    # Known lower bounds and results for 3x5x5:
    # The rank of matrix multiplication [3,5,5] is known to be 75 (schoolbook) 
    # but can potentially be lower. Let's try to find a better decomposition.
    
    # For small dimensions, we can try random initializations and optimization.
    # The tensor T[i,j,k] = 1 if the product (A*B)[i,k] uses A[i,j]*B[j,k].
    # We want T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product sum)
    
    # Let's try a few random restarts with rank slightly below 75 to see if we can find a solution.
    # If not, return the schoolbook solution.
    
    def tensor_residual(rank, U, V, W):
        """Compute the residual of the tensor decomposition."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[i*m+j, j*p+k, k*n+i] should be 1 for all valid i,j,k
        # Actually, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # We need to check all (i,j,k) triples
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residual += (val - 1.0) ** 2
        return residual
    
    def flatten_params(rank):
        """Flatten U, V, W into a single vector."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    def objective(x):
        rank = len(x) // (n*m + m*p + p*n)
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = x[(n*m+m*p)*rank:].reshape(p*n, rank)
        return tensor_residual(rank, U, V, W)
    
    # Try a few ranks below 75
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try rank 74, 73, ... down to maybe 70
    for rank in range(best_rank - 1, best_rank - 6, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            random.seed(42 + rank + attempt)
            np.random.seed(42 + rank + attempt)
            
            # Initialize with small random values
            x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.1
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < 1e-6:  # Found a good solution
                    best_rank = rank
                    # Reconstruct U, V, W
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
                    W = result.x[(n*m+m*p)*rank:].reshape(p*n, rank)
                    
                    # Convert to list of lists with rounded values (half-integers)
                    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
                    
                    # Round to nearest 0.5
                    for mat in [U_list, V_list, W_list]:
                        for i in range(len(mat)):
                            for j in range(len(mat[i])):
                                mat[i][j] = round(mat[i][j] * 2) / 2
                    
                    # Verify the solution
                    if tensor_residual(rank, 
                                       np.array(U_list), 
                                       np.array(V_list), 
                                       np.array(W_list)) < 1e-4:
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception as e:
                pass
    
    # Fall back to schoolbook decomposition
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
# EVOLVE-BLOCK-END
