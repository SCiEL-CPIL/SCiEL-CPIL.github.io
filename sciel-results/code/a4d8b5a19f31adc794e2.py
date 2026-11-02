"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook initialization
    rank0 = n * m * p
    
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact. Returns True if valid."""
        r = U.shape[1]
        # Vectorized check: for each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # This should equal 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the sum
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Verify schoolbook
    if verify(U0, V0, W0):
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank0
    else:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0
    
    # Local search: try to eliminate redundant columns from schoolbook
    # Strategy: for each column, check if it can be removed by adjusting others.
    # Use a greedy approach: try removing one column at a time.
    
    def try_remove_column(U, V, W, col_idx, tol=1e-9):
        """Try to remove column col_idx. Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
        r = U.shape[1]
        if r <= 1:
            return False, None, None, None
        
        # Create copies without the column
        mask = np.ones(r, dtype=bool)
        mask[col_idx] = False
        U_new = U[:, mask].copy()
        V_new = V[:, mask].copy()
        W_new = W[:, mask].copy()
        
        # Check if the remaining columns still form a valid decomposition
        if verify(U_new, V_new, W_new, tol=tol):
            return True, U_new, V_new, W_new
        
        # Try to adjust the remaining columns to compensate
        # This is a hard problem, so use a simple heuristic:
        # Try small perturbations of the remaining columns
        
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            U_try = U_new.copy()
            V_try = V_new.copy()
            W_try = W_new.copy()
            
            # Add small random perturbation
            U_try += np.random.randn(nm, r-1) * 0.01
            V_try += np.random.randn(mp, r-1) * 0.01
            W_try += np.random.randn(pn, r-1) * 0.01
            
            if verify(U_try, V_try, W_try, tol=1e-6):
                return True, U_try, V_try, W_try
        
        return False, None, None, None
    
    # Greedy column elimination
    U_cur, V_cur, W_cur = U0.copy(), V0.copy(), W0.copy()
    rank_cur = rank0
    
    # Try to eliminate columns in random order
    col_order = np.random.permutation(rank0)
    for col_idx in col_order:
        if time.time() > deadline - 10:
            break
        
        success, U_new, V_new, W_new = try_remove_column(U_cur, V_cur, W_cur, col_idx, tol=1e-9)
        if success:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            rank_cur = U_cur.shape[1]
            if rank_cur < best_rank:
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                best_rank = rank_cur
    
    # Final verification
    if verify(best_U, best_V, best_W, tol=1e-9):
        return to_list(best_U, best_V, best_W)
    else:
        # Return schoolbook if search failed
        return to_list(U0, V0, W0)
# EVOLVE-BLOCK-END
