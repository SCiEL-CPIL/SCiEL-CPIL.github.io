import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Check deadline - if we're close, return immediately
    if time.time() - start_time > deadline - 5.0:
        return best_result
    
    # Try to search for a lower rank decomposition
    # For M_{2,3,8}, attempt to find rank < 48
    # Use randomized approach: try random low-rank decompositions
    
    np.random.seed(42)
    
    # Try various ranks from 47 down to some minimum
    # For each rank, try multiple random initializations
    min_rank_to_try = max(1, rank - 20)  # Try ranks from 47 down to 28
    
    for r in range(rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > deadline - 10.0:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Generate random U and V matrices
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            
            # For each rank-1 component, we need W[k*n+i] such that
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if (i,j,k) is valid
            # The tensor T[i,j,k] = 1 for all valid (i,j,k)
            
            # We need to solve for W given U and V
            # T = sum_r U[:,i*m+j] * V[:,j*p+k] * W[:,k*n+i]
            
            # This is a linear system. Let's set up the equations:
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # We have p*n = 16 equations for W variables (but W has p*n*r variables)
            # Actually, W has p*n rows and r columns, so p*n*r unknowns
            # But the equations couple them in a complex way
            
            # Alternative: use the fact that for matrix multiplication,
            # we can think of it as: C[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: T[i,j,k] = 1
            
            # Let's try a different approach: random W and check if the
            # resulting tensor matches
            
            # Generate random W
            W_rand = np.random.randn(p * n, r)
            
            # Compute the tensor from the decomposition
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            tensor_computed = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_computed[i,j,k] = np.dot(
                            U_rand[i*m+j, :], 
                            V_rand[j*p+k, :] * W_rand[k*n+i, :]
                        )
            
            # Check if tensor_computed is close to all-ones tensor
            if np.allclose(tensor_computed, 1.0, atol=1e-10):
                # Found a valid decomposition!
                best_rank = r
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                break
        
        if best_rank == r:
            break  # Found the best possible for this rank
    
    return best_result
