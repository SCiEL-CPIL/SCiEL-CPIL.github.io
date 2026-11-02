import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Now build the true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_reduce_rank(U, V, W, target_rank, n, m, p, time_limit=60):
    """
    Try to reduce the rank from current to target_rank by zeroing out columns
    and optimizing the remaining ones.
    Returns (U, V, W, success)
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, False
    
    # Try each possible set of columns to zero out
    # For efficiency, try zeroing out columns that are most "redundant"
    # Start with the last few columns (trivial ones)
    
    # Strategy: try removing one column at a time, starting from the end
    # and also try removing columns from Strassen blocks
    
    # First, try simply zeroing the last (current_rank - target_rank) columns
    # and see if optimization can fix it
    
    num_to_remove = current_rank - target_rank
    
    # Try different removal patterns
    removal_patterns = []
    
    # Pattern 1: remove last num_to_remove columns
    removal_patterns.append(list(range(current_rank - num_to_remove, current_rank)))
    
    # Pattern 2: remove one from each trivial block (if applicable)
    # The trivial block is the last 4 columns (indices 28-31)
    if num_to_remove <= 4:
        # Try removing from the trivial block
        trivial_indices = list(range(28, 32))
        for i in range(len(trivial_indices)):
            removal_patterns.append([trivial_indices[i]])
    
    # Pattern 3: try removing from different Strassen blocks
    # Strassen blocks are at indices 0-6, 7-13, 14-20, 21-27
    strassen_blocks = [list(range(0,7)), list(range(7,14)), list(range(14,21)), list(range(21,28))]
    for block in strassen_blocks:
        for idx in block:
            removal_patterns.append([idx])
    
    # Try each removal pattern
    for removal in removal_patterns:
        if time.time() - start_time > time_limit:
            break
        if len(removal) != num_to_remove:
            continue
            
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        
        for idx in removal:
            U_test[:, idx] = 0
            V_test[:, idx] = 0
            W_test[:, idx] = 0
        
        # Now optimize the non-zero columns
        # Flatten the non-zero parts
        keep_mask = np.ones(current_rank, dtype=bool)
        for idx in removal:
            keep_mask[idx] = False
        
        keep_indices = np.where(keep_mask)[0]
        
        # Extract the free parameters
        U_free = U_test[:, keep_indices].copy()
        V_free = V_test[:, keep_indices].copy()
        W_free = W_test[:, keep_indices].copy()
        
        # Flatten
        x0 = np.concatenate([U_free.flatten(), V_free.flatten(), W_free.flatten()])
        
        def objective(x):
            U_f = x[:U_free.size].reshape(U_free.shape)
            V_f = x[U_free.size:U_free.size+V_free.size].reshape(V_free.shape)
            W_f = x[U_free.size+V_free.size:].reshape(W_free.shape)
            
            # Compute tensor
            M = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
            
            # True tensor
            T_true = np.zeros((n * m, m * p, p * n))
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a * m + b
                        j = b * p + c
                        k = c * n + a
                        T_true[i, j, k] = 1.0
            
            residual = M - T_true
            # Use L2 norm squared for smoothness
            return np.sum(residual**2)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                # Success! Extract the solution
                U_f = result.x[:U_free.size].reshape(U_free.shape)
                V_f = result.x[U_free.size:U_free.size+V_free.size].reshape(V_free.shape)
                W_f = result.x[U_free.size+V_free.size:].reshape(W_free.shape)
                
                U_new = np.zeros((n*m, target_rank))
                V_new = np.zeros((m*p, target_rank))
                W_new = np.zeros((p*n, target_rank))
                
                for new_idx, old_idx in enumerate(keep_indices):
                    U_new[:, new_idx] = U_f[:, new_idx - keep_indices[0]] if False else U_f[:, np.where(keep_indices == old_idx)[0][0]]
                    V_new[:, new_idx] = V_f[:, np.where(keep_indices == old_idx)[0][0]]
                    W_new[:, new_idx] = W_f[:, np.where(keep_indices == old_idx)[0][0]]
                
                # Verify
                M_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                T_true = np.zeros((n * m, m * p, p * n))
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            i = a * m + b
                            j = b * p + c
                            k = c * n + a
                            T_true[i, j, k] = 1.0
                
                if np.max(np.abs(M_check - T_true)) < 1e-8:
                    return U_new, V_new, W_new, True
                    
        except Exception:
            continue
    
    return U, V, W, False

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    # Verify the rank-32 solution
    residual = _compute_residual(U32, V32, W32, n, m, p)
    
    # Try to reduce to rank 31
    time_remaining = 120 - (time.time() - start_time) - 10  # 10s safety margin
    if residual < 1e-10 and time_remaining > 30:
        U31, V31, W31, success = _try_reduce_rank(U32, V32, W32, 31, n, m, p, time_limit=time_remaining)
        if success:
            return {"U": U31.tolist(), "V": V31.tolist(), "W": W31.tolist()}
    
    # Return the verified rank-32 solution
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
