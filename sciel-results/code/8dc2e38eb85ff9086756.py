import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
        
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    if not np.allclose(T_computed, T_target, atol=1e-9):
        return False
        
    return True

def try_reduce_with_patterns(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding structured patterns that allow term elimination.
    Returns (U, V, W) with reduced rank if successful, else original.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = U.shape[1]
    
    # Try various structured reductions
    # Pattern 1: For fixed i, try to combine terms across j
    # Pattern 2: For fixed j, try to combine terms across k
    # Pattern 3: For fixed k, try to combine terms across i
    
    best_U, best_V, best_W = current_U, current_V, current_W
    best_rank = current_rank
    
    # Try removing one term at a time and checking if the system remains solvable
    for trial in range(current_rank):
        if time.time() > deadline:
            break
            
        # Skip if already removed
        if np.all(current_U[:, trial] == 0):
            continue
            
        # Try removing this term
        test_U = np.delete(current_U, trial, axis=1)
        test_V = np.delete(current_V, trial, axis=1)
        test_W = np.delete(current_W, trial, axis=1)
        
        # Check if the remaining terms can still span the target tensor
        # This is a complex check, so we'll use a heuristic:
        # Try to see if we can adjust other terms to compensate
        
        # For now, just check if removing this term still gives a valid decomposition
        if verify_decomposition(test_U, test_V, test_W, n, m, p):
            current_U, current_V, current_W = test_U, test_V, test_W
            current_rank -= 1
            best_U, best_V, best_W = current_U, current_V, current_W
            best_rank = current_rank
            # Restart from beginning since indices changed
            trial = -1
    
    return best_U, best_V, best_W, best_rank

def local_optimization(U, V, W, n, m, p, deadline):
    """
    Perform local optimization to find better factor values.
    Uses gradient-free optimization on the factor matrices.
    """
    U_opt = U.astype(np.float64).copy()
    V_opt = V.astype(np.float64).copy()
    W_opt = W.astype(np.float64).copy()
    
    rank = U.shape[1]
    best_rank = rank
    
    # Try to zero out entries that are close to zero
    for col in range(rank):
        if time.time() > deadline:
            break
            
        # Check if this column can be made zero
        col_norm = np.linalg.norm(U_opt[:, col]) + np.linalg.norm(V_opt[:, col]) + np.linalg.norm(W_opt[:, col])
        if col_norm < 1e-10:
            continue
        
        # Try setting this column to zero and see if we can compensate
        saved_U = U_opt[:, col].copy()
        saved_V = V_opt[:, col].copy()
        saved_W = W_opt[:, col].copy()
        
        U_opt[:, col] = 0
        V_opt[:, col] = 0
        W_opt[:, col] = 0
        
        # Check if this still works (it won't unless other terms compensate)
        if verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
            best_rank -= 1
        else:
            # Restore
            U_opt[:, col] = saved_U
            V_opt[:, col] = saved_V
            W_opt[:, col] = saved_W
    
    # Try to merge columns that are similar
    for col1 in range(rank):
        if time.time() > deadline:
            break
        for col2 in range(col1 + 1, rank):
            # Check if columns are proportional
            if np.allclose(U_opt[:, col1], U_opt[:, col2]) and \
               np.allclose(V_opt[:, col1], V_opt[:, col2]) and \
               np.allclose(W_opt[:, col1], W_opt[:, col2]):
                # These can be merged
                U_opt[:, col1] += U_opt[:, col2]
                V_opt[:, col1] += V_opt[:, col2]
                W_opt[:, col1] += W_opt[:, col2]
                U_opt[:, col2] = 0
                V_opt[:, col2] = 0
                W_opt[:, col2] = 0
                best_rank -= 1
                break
    
    return U_opt, V_opt, W_opt, best_rank

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110  # Safety margin before 120s limit
    
    # Start with trivial decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try pattern-based reduction
    if time.time() < deadline - 20:
        U2, V2, W2, rank2 = try_reduce_with_patterns(U, V, W, n, m, p, deadline)
        if rank2 < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = rank2
    
    # Try local optimization
    if time.time() < deadline - 10:
        U3, V3, W3, rank3 = local_optimization(best_U, best_V, best_W, n, m, p, deadline)
        if rank3 < best_rank:
            best_U, best_V, best_W = U3, V3, W3
            best_rank = rank3
    
    # Final verification
    result = {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
    
    if verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # Fallback to original if something went wrong
        return {
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }

if __name__ == '__main__':
    res = construct()
    print(f"Rank: {len(res['U'][0]) if res['U'] else 0}")
