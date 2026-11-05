"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Condition: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Vectorized verification."""
        rank = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # For each (i,j,k), compute sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # This is a bit tricky to vectorize fully, but we can do it
        result = np.zeros((n, m, p))
        for c in range(rank):
            # U3[:, :, c] has shape (n, m)
            # V3[:, :, c] has shape (m, p)
            # W3[:, :, c] has shape (p, n)
            # We need: for each i,j,k: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            # This is like an element-wise product after appropriate reshaping
            # U3[:,:,c] is (n,m), V3[:,:,c] is (m,p), W3[:,:,c] is (p,n)
            # We can compute: np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            result += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
        return np.allclose(result, 1.0, atol=1e-9)
    
    def make_half_integer(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find lower rank decompositions
    best_rank = n * m * p  # 45
    best_decomp = schoolbook()
    
    # Try ranks from a lower bound up to best_rank
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15? 
    # Actually, the rank must be at least the number of "essential" terms
    # For safety, start from 20
    min_rank_to_try = 18
    
    random.seed(42)
    
    for rank in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if rank < 30 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with half-integer values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            if verify_fast(U, V, W):
                best_rank = rank
                best_decomp = (U, V, W)
                break  # Found valid decomposition at this rank, no need to try higher
        
        if best_rank == rank:
            break  # Found the lowest rank we've been searching for
    
    # Convert to lists for output
    U, V, W = best_decomp
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are exactly half-integers
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = round(U_list[i][j] * 2) / 2
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = round(V_list[i][j] * 2) / 2
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = round(W_list[i][j] * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
