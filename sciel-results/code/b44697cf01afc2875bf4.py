"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strategy: Split the problem into 2x3x3 and 2x3x2
    # Find low-rank decompositions for each part and concatenate
    
    # Part 1: 2x3x3
    U1, V1, W1, rank1 = _find_sub_decomposition(n, m, 3, 18, 14, deadline, start_time, seed=101)
    # Part 2: 2x3x2
    U2, V2, W2, rank2 = _find_sub_decomposition(n, m, 2, 12, 10, deadline, start_time, seed=202)
    
    # Combine
    if U1 is not None and U2 is not None:
        total_rank = rank1 + rank2
        if total_rank < best_rank:
            U_comb = np.hstack([U1, U2])
            V_comb = np.hstack([V1, V2])
            # W needs careful concatenation. 
            # W1 has shape (3*n, rank1). W2 has shape (2*n, rank2).
            # The full W has shape (5*n, total_rank).
            # W[k*n+i, c] for k in 0..2 comes from W1.
            # W[k*n+i, c] for k in 3..4 comes from W2 (shifted index).
            W_comb = np.zeros((p * n, total_rank))
            W_comb[:3*n, :rank1] = W1
            # For the second part, k in 3,4 corresponds to indices 3*n, 3*n+1...
            # In the sub-problem 2x3x2, the k index is 0,1.
            # So W2[k_sub*n + i, c] maps to W_comb[(k_sub+3)*n + i, c]
            W_comb[3*n:5*n, rank1:] = W2
            
            best_rank = total_rank
            best_result = {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }

    # If no improvement found, try a global search with tighter constraints
    if best_rank == rank_school:
         # Fallback to a more aggressive random search on the full problem
         # but with smaller coefficient range and more attempts
         for target_rank in range(25, rank_school):
             if time.time() > deadline - 10:
                 break
             np.random.seed(42 + target_rank)
             for attempt in range(100):
                 if time.time() > deadline - 5:
                     break
                 U = np.random.randint(-4, 5, size=(n*m, target_rank)) / 2.0
                 V = np.random.randint(-4, 5, size=(m*p, target_rank)) / 2.0
                 W = np.random.randint(-4, 5, size=(p*n, target_rank)) / 2.0
                 if _verify_decomposition_fast(U, V, W, n, m, p):
                     best_rank = target_rank
                     best_result = {
                         "U": U.tolist(),
                         "V": V.tolist(),
                         "W": W.tolist()
                     }
                     break
             if best_rank == target_rank:
                 break

    return best_result

def _find_sub_decomposition(n, m, p_sub, max_rank, min_rank, deadline, start_time, seed):
    """
    Finds a decomposition for n x m x p_sub.
    Returns U, V, W, rank or (None, None, None, max_rank) if not found.
    """
    # If time is tight, return schoolbook
    if time.time() > deadline - 5:
        return _schoolbook(n, m, p_sub)

    np.random.seed(seed)
    
    # Try ranks from min_rank to max_rank
    for target_rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # More attempts for lower ranks
        attempts = 500 if target_rank < 16 else 200
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Random half-integers
            U = np.random.randint(-4, 5, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-4, 5, size=(m*p_sub, target_rank)) / 2.0
            W = np.random.randint(-4, 5, size=(p_sub*n, target_rank)) / 2.0
            
            if _verify_decomposition_fast(U, V, W, n, m, p_sub):
                return U, V, W, target_rank
                
    return _schoolbook(n, m, p_sub)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Vectorized verification using NumPy.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    
    # Reshape for easier broadcasting
    # U: (n, m, rank)
    U_r = U.reshape(n, m, rank)
    # V: (m, p, rank)
    V_r = V.reshape(m, p, rank)
    # W: (p, n, rank)
    W_r = W.reshape(p, n, rank)
    
    # We want to compute for each i, j, k:
    # sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # This is a tensor contraction.
    # Let's iterate over i and j to reduce memory, or use einsum.
    # np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check if all entries are 1.0
        return np.allclose(T, 1.0, atol=1e-9)
    except Exception:
        return False

# EVOLVE-BLOCK-END
