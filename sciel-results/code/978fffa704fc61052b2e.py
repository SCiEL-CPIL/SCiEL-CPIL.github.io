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
    
    def verify_fast(U, V, W):
        """Vectorized verification."""
        rank = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # For each (i,j,k), compute sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        result = np.zeros((n, m, p))
        for c in range(rank):
            # U3[:,:,c] has shape (n, m)
            # V3[:,:,c] has shape (m, p)
            # W3[:,:,c] has shape (p, n)
            # We need: for each i,j,k: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
            result += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
        return np.allclose(result, 1.0, atol=1e-9)
    
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
    
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15? 
    # For safety, start from 18
    min_rank_to_try = 18
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Border rank perturbation
    # Start from a known structure and try to reduce rank by introducing linear dependencies
    # We'll try systematic searches with different initialization strategies
    
    for rank in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with different strategies
        max_attempts = 100 if rank < 30 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Strategy 1: Random half-integer values
            if attempt % 3 == 0:
                U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
                V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
                W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Strategy 2: Sparse random with more zeros
            elif attempt % 3 == 1:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    # Randomly select a few non-zero entries
                    u_idx = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    v_idx = np.random.choice(m*p, size=min(3, m*p), replace=False)
                    w_idx = np.random.choice(p*n, size=min(3, p*n), replace=False)
                    U[u_idx, c] = np.random.choice([-1, 1])
                    V[v_idx, c] = np.random.choice([-1, 1])
                    W[w_idx, c] = np.random.choice([-1, 1])
            
            # Strategy 3: Structured initialization based on block patterns
            else:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for c in range(rank):
                    # Assign based on pattern
                    i_block = c // (m*p)
                    j_block = (c % (m*p)) // p
                    k_block = c % p
                    if i_block < n and j_block < m and k_block < p:
                        U[i_block*m+j_block, c] = 1
                        V[j_block*p+k_block, c] = 1
                        W[k_block*n+i_block, c] = 1
                    else:
                        # Add random perturbation
                        U[np.random.randint(n*m), c] = np.random.choice([-1, -0.5, 0.5, 1])
                        V[np.random.randint(m*p), c] = np.random.choice([-1, -0.5, 0.5, 1])
                        W[np.random.randint(p*n), c] = np.random.choice([-1, -0.5, 0.5, 1])
            
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
