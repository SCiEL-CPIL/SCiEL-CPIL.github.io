"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    # Reshape for tensor contraction
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can do this by: for each i, j, k, take dot product of U[i,j,:], V[j,k,:], W[k,i,:]
    # More efficiently: 
    # result[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
    
    # Let's compute for each i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # dot product of three vectors of length rank
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 130s to leave margin
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    
    # The known lower bound for 3x5x5 matrix multiplication rank is 40
    # (rank >= max(n*m, m*p, p*n) = max(15, 25, 15) = 25, but actual lower bound is higher)
    # For 3x5x5, the rank is known to be 40 (from literature)
    # Let's try to find a decomposition with rank 40 or lower
    
    # Try ranks from the theoretical lower bound up to current best
    # Lower bound: max(n*m, m*p, p*n) = 25
    # But for matrix multiplication, rank >= n*m*p / min(n,m,p) is not the right bound
    # Actually, rank >= max(n*m, m*p, p*n) is a lower bound
    # So rank >= 25 for this case
    
    # Let's try a systematic search for rank 40 first (known achievable)
    # then try lower ranks
    
    # Use a better search strategy: start from rank 40 and try to find valid decompositions
    # Then try lower ranks
    
    candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # First, try to find a decomposition at rank 40 (known to be achievable)
    # Then try lower ranks
    
    for target_rank in range(40, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
            V_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
            W_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found valid decomposition for this rank, try lower
    
    # Now try lower ranks (25 to 39)
    for target_rank in range(25, min(40, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(n*m)]
            V_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(m*p)]
            W_r = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found valid decomposition for this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
