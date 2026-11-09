"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p
    rank_schoolbook = n * m * p
    
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a lower rank decomposition using numerical methods
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) is a valid triple, 0 otherwise
    # Actually, for matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has T[i,j,k] = 1 for all i,j,k (it's just the multiplication tensor)
    # Wait, let me reconsider. The matrix multiplication tensor T_{ijk} where:
    # i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1), k indexes columns of B (0..p-1)
    # T_{ijk} = 1 for all i,j,k
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a 3-tensor decomposition problem. Let's try rank < 42
    
    # Use a simple approach: try to find rank 41, 40, ... decompositions
    # For small improvements, we can use the fact that some entries might be zero
    
    # Actually, for 2x3x7, the known lower bound is 42 (schoolbook), and it's not known if lower rank is possible
    # Let's try a few ranks and see if we can find a solution
    
    # We'll use a simple iterative approach
    for target_rank in range(rank_schoolbook - 1, 35, -1):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        np.random.seed(42)
        
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use alternating least squares
            for iteration in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                        # This is a linear system in U[i*m+j, :]
                        # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                        # We need sum_r U[i*m+j, r] * A_k[r] = 1 for all k
                        # So U[i*m+j, :] * A^T = 1^T
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A^T * U[i*m+j, :]^T = 1
                        sol, _ = nnls(A.T, np.ones(p))
                        U[i * m + j, :] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        sol, _ = nnls(A.T, np.ones(n))
                        V[j * p + k, :] = sol
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        sol, _ = nnls(A.T, np.ones(m))
                        W[k * n + i, :] = sol
            
            # Check if we found a valid solution
            max_error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        max_error = max(max_error, abs(val - 1))
            
            if max_error < 1e-6:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_result["U"] != U_sb:
            break
    
    return best_result
# EVOLVE-BLOCK-END
