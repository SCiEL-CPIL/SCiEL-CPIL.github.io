"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    which is 1 if (i,j,k) is a valid index (always is) and 0 otherwise.
    Actually, the matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    and the decomposition should satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against identity: T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, no. The matrix multiplication tensor is defined as:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # Actually the standard matrix multiplication tensor T_{i,j,k} = delta_{i,j,k} is not right.
    # 
    # Let me re-read the problem. The tensor for matrix multiplication C = A*B where
    # A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has components T[i,j,k] such that the trilinear form is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^T[i,k] = sum_{i,k} C[i,k] C^T[i,k]
    # So T[i,j,k] = 1 for all valid i, j, k.
    #
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Use numpy for efficient verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # This is a bit tricky with einsum
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _random_decomposition(rank, n, m, p, seed=None):
    """Generate a random decomposition and check if it's valid."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate random matrices with entries in {-1, 0, 1} or small integers
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                U[i*m+j, c] = random.choice([-1, 0, 1])
        for j in range(m):
            for k in range(p):
                V[j*p+k, c] = random.choice([-1, 0, 1])
        for k in range(p):
            for i in range(n):
                W[k*n+i, c] = random.choice([-1, 0, 1])
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_rank(rank, n, m, p, max_attempts=100, seed_offset=0):
    """Try to find a valid decomposition of given rank."""
    for attempt in range(max_attempts):
        result = _random_decomposition(rank, n, m, p, seed=seed_offset + attempt)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Also rank >= n+m+p-2 = 3+4+5-2 = 10 (but this is a weak bound)
    # For 3x4x5, the known optimal rank is likely around 20-25
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    # Actually, a better lower bound: rank >= max(n*m, m*p, p*n) is not always tight
    # Let's try from a reasonable range
    
    # Try ranks from 20 to 60, with more attempts for lower ranks
    for rank in range(20, min(60, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        # Use more attempts for lower ranks
        attempts = 50 if rank < 25 else 30
        result = _try_rank(rank, n, m, p, max_attempts=attempts, seed_offset=rank * 1000)
        if result is not None:
            best_result = result
            best_rank = rank
            # Found a valid decomposition, continue searching for lower rank
            # But we're going upward, so this is the first valid one at this rank
            # Actually we should continue to see if we can find even lower, but we're going up
            # So break after finding the first valid one at the lowest rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
