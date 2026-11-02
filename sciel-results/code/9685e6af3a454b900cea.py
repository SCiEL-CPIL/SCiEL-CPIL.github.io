"""Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix."""
DIMENSIONS = [3, 4, 4]
import numpy as np
import time
import random
from itertools import product, combinations

# EVOLVE-BLOCK-START
def _matmul_tensor(n, m, p):
    """Return the matrix multiplication tensor T as a 3D array of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matmul tensor for DIMENSIONS.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Returns (is_valid, rank).
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # For each c: U[:,c] * V[:,c] * W[:,c]  (outer product)
    # Accumulate: T = sum_c outer(U[:,c], V[:,c]) * W[:,c][None, None, :]
    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for c in range(rank):
        T_reconstructed += np.outer(U_arr[:, c], V_arr[:, c]) * W_arr[:, c][None, None, :]
    
    T_target = _matmul_tensor(n, m, p)
    if not np.allclose(T_reconstructed, T_target, atol=1e-10, rtol=1e-10):
        return False, rank
    return True, rank

def _to_int_or_half_int(val):
    """Round to nearest half-integer and return as int (twice the value) for exactness."""
    return int(round(val * 2))

def _from_twice_int(twice_val):
    return twice_val / 2.0

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    U = [row[:] for row in U_sb]
    V = [row[:] for row in V_sb]
    W = [row[:] for row in W_sb]
    current_rank = rank_school
    best_valid = (True, U, V, W)
    
    T_target = _matmul_tensor(n, m, p)
    
    # Strategy: try to reduce rank by setting one column to zero and compensating,
    # or by trying low-rank factorizations with integer/half-integer entries.
    
    # Approach 1: Random search for low-rank decompositions with entries in {-1, 0, 1, 0.5, ...}
    # Try ranks from a reasonable lower bound up to current_rank - 1
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) is not tight
    # Actually, the known lower bounds for 3x4x4 are not well established. Let's try from 30 to 47.
    
    # Use a more structured search: try rank from current_rank-1 down to 20
    # But this is computationally expensive. Instead, let's try a few random restarts.
    
    np.random.seed(42)
    random.seed(42)
    
    entries = [-1.0, -0.5, 0.0, 0.5, 1.0]  # half-integer entries
    
    # Try to find a decomposition of rank r for various r
    # We'll use a simple approach: for each candidate rank, do a few random trials
    for target_rank in range(47, 29, -1):
        if time.time() > deadline - 5:
            break
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Generate random matrices with entries from the allowed set
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            for idx in range(n * m * target_rank):
                U_try.flat[idx] = entries[random.randint(0, len(entries) - 1)]
            for idx in range(m * p * target_rank):
                V_try.flat[idx] = entries[random.randint(0, len(entries) - 1)]
            for idx in range(p * n * target_rank):
                W_try.flat[idx] = entries[random.randint(0, len(entries) - 1)]
            
            T_recon = np.zeros((n * m, m * p, p * n))
            for c in range(target_rank):
                T_recon += np.outer(U_try[:, c], V_try[:, c]) * W_try[:, c][None, None, :]
            
            if np.allclose(T_recon, T_target, atol=1e-10, rtol=1e-10):
                U_list = [[float(x) for x in row] for row in U_try]
                V_list = [[float(x) for x in row] for row in V_try]
                W_list = [[float(x) for x in row] for row in W_try]
                valid, _ = _verify(U_list, V_list, W_list)
                if valid:
                    current_rank = target_rank
                    U, V, W = U_list, V_list, W_list
                    best_valid = (True, U, V, W)
                    break
        else:
            continue
        break
        # Found a valid decomposition at this rank, try lower
    
    # Return the best valid decomposition
    if best_valid[0]:
        return {"U": best_valid[1], "V": best_valid[2], "W": best_valid[3]}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
