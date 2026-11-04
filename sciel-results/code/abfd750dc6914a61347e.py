"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_full(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the FULL matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct the full tensor
    # T[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
    # We need T[u, v, w] = 1 if (u, v, w) corresponds to a valid (i, j, k)
    # and 0 otherwise.
    
    # Map indices
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    T_recon = np.einsum('uc,vc,wc->uvw', U_arr, V_arr, W_arr)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=tol):
        return False
    return True

def _validate_partial(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check the 40 non-zero entries (as in the prompt's snippet)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integer(arr, tol=1e-9):
    for row in arr:
        for val in row:
            if not _is_half_integer(val, tol):
                return False
    return True

def _try_rank_search(n=2, m=4, p=5, start_rank=20, max_rank=40, time_limit=100):
    """
    Search for a valid decomposition with rank in [start_rank, max_rank].
    Uses random initialization and local search.
    """
    start_time = time.time()
    best_result = None
    
    for r in range(start_rank, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with small integers/half-integers
            # Values: -1, -0.5, 0, 0.5, 1
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U = [[random.choice(vals) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(p*n)]
            
            # Check if it works immediately
            if _validate_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
            # Local search: flip random entries
            # We need to minimize the error.
            # For simplicity, we just do a few random flips and check.
            # This is inefficient but might find a solution if rank is close to optimal.
            
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random matrix, row, col
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    rows = n*m
                elif mat_idx == 1:
                    mat = V
                    rows = m*p
                else:
                    mat = W
                    rows = p*n
                    
                row = random.randint(0, rows - 1)
                col = random.randint(0, r - 1)
                
                # Flip the value
                old_val = mat[row][col]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                    
                mat[row][col] = new_val
                
                if _validate_full(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                
                # Backtrack if error increased? 
                # For simplicity, we don't calculate error explicitly here, 
                # just check validity. This is a random walk.
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Try to find a low-rank decomposition (full tensor check)
    # Start from the lower bound
    result = _try_rank_search(n, m, p, start_rank=20, max_rank=39, time_limit=100)
    
    if result is not None:
        # Verify again
        assert _validate_full(result["U"], result["V"], result["W"], n, m, p), "Full validation failed"
        assert _all_half_integer(result["U"]), "U invalid entries"
        assert _all_half_integer(result["V"]), "V invalid entries"
        assert _all_half_integer(result["W"]), "W invalid entries"
        return result
        
    # 2. Fallback to Schoolbook (Rank 40)
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
                
    # Verify schoolbook
    assert _validate_full(U, V, W, n, m, p), "Schoolbook full validation failed"
    assert _all_half_integer(U), "U invalid entries"
    assert _all_half_integer(V), "V invalid entries"
    assert _all_half_integer(W), "W invalid entries"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
