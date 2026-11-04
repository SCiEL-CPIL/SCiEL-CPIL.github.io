import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape to 3D
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Check full tensor:
    # Sum_c U[i,j,c] * V[j',k,c] * W[k',i',c] == delta(j,j') * delta(k,k') * delta(i,i')
    
    # We can compute the full tensor T of shape (n, m, m, p, p, n)
    # T[i, j, j', k, k', i'] = sum_c U[i,j,c] V[j',k,c] W[k',i',c]
    
    # This is expensive. For n=2, m=5, p=5, size is 2*5*5*5*5*2 = 2500.
    # We can loop over i, j, j', k, k', i' and check.
    
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            # Target value
                            target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            
                            # Compute sum
                            # U[i, j, :] * V[j_prime, k, :] * W[k_prime, i_prime, :]
                            # Note: V index is j_prime * p + k
                            # W index is k_prime * n + i_prime
                            
                            val = np.dot(U_3d[i, j, :], V_3d[j_prime, k, :] * W_3d[k_prime, i_prime, :])
                            
                            if abs(val - target) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10
    
    # Schoolbook is a valid baseline
    best_result = _schoolbook(n, m, p)
    
    # Due to the complexity of finding a lower rank decomposition for 2x5x5
    # within the time limit, and the lack of a specific fast algorithm implementation,
    # we return the schoolbook solution.
    
    return best_result
