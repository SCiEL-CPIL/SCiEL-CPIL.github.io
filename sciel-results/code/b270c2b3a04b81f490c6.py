"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
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
    
    # Verify a decomposition is exact
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Convert to numpy arrays for efficient verification
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = 1 if and only if (i,j,k) is a valid index
        # For matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check all entries for exact verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # For 2x3x10, we can potentially use rank < 60 by sharing computations
    # Strategy: Use bilinear structure and exploit sharing across the p=10 dimension
    
    best_result = None
    best_rank = n * m * p
    
    # Try various target ranks with structured random search
    for target_rank in [40, 45, 50, 55, 58, 59]:
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random half-integers
            U_rand = np.random.randint(0, 3, (n*m, target_rank)) * 0.5
            V_rand = np.random.randint(0, 3, (m*p, target_rank)) * 0.5
            W_rand = np.random.randint(0, 3, (p*n, target_rank)) * 0.5
            
            # Convert to lists
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    # Return best result found, or schoolbook if nothing better
    if best_result is not None and best_rank < n * m * p:
        return best_result
    else:
        return make_schoolbook()
