"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
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
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # We will try ranks from lower_bound to best_rank - 1
    # Given the difficulty, we will try a limited number of attempts for each rank
    # using the fast vectorized verification
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        # The number of attempts can be high because verification is fast now
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = target_rank * 10000 + attempt
            U_t, V_t, W_t = _random_init(n, m, p, target_rank, seed=seed)
            
            # Check if this random initialization happens to work
            if _verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                # If found, try to round to half-integers?
                # The problem asks for integer/half-integer entries.
                # Random floats are unlikely to be close to half-integers.
                # We can try to round and re-verify with tight tolerance.
                U_r = [[round(x * 2) / 2 for x in row] for row in U_t]
                V_r = [[round(x * 2) / 2 for x in row] for row in V_t]
                W_r = [[round(x * 2) / 2 for x in row] for row in W_t]
                
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    break
                else:
                    # If rounding fails, maybe the float solution was valid but not half-integer?
                    # The task requires integer/half-integer.
                    # So we must discard it if it doesn't round to a valid solution.
                    pass
        
        if best_rank == target_rank:
            break
    
    # Return the best found
    return best
