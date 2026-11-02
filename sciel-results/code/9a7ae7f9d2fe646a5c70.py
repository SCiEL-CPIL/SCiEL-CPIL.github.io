"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matmul tensor for 2x5x5.
    U: (n*m, r), V: (m*p, r), W: (p*n, r) with W[k*n+i, c]
    Product: for each c, U[:,c] V[:,c] gives (n*m) x (m*p) matrix.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a
    valid matmul term... wait, the tensor decomposition means:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    where T[i,j,k] = 1 for all i,j,k (the matmul tensor has all ones).
    Actually for matmul (2x5)*(5x5), the tensor T has T[i,j,k] = 1 for all i in [0,2), j in [0,5), k in [0,5).
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Build the tensor from the factors
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # All entries should be exactly 1
    if T.shape != (n, m, p):
        return False
    return bool(np.all(T == 1))

def _try_rank_reduction():
    """Try to find a lower rank decomposition. For 2x5x5, schoolbook is 50.
    We attempt ranks 49 down to 45 using randomized search with integer/half-integer entries."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # The matmul tensor for (2x5)*(5x5) has all entries = 1
    # We need: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # for all i in [0,2), j in [0,5), k in [0,5)
    
    # Strategy: Start from schoolbook and try to merge/replace terms.
    # For rank reduction by 1, we need to express one basis element as a combination
    # of others, or find a more efficient structure.
    
    # Known bounds: For 2xm x mxn type problems, the rank can sometimes be reduced.
    # For 2x5x5, let's try a structured approach.
    
    # One approach: Use the fact that we can group terms.
    # Consider pairs of k values for the same (i,j).
    # T[i,j,k] = 1 for all k.
    
    # Let's try a direct randomized local search for small rank reductions.
    # Initialize with schoolbook, then try to set some entries to non-trivial values
    # to satisfy the constraints with fewer columns.
    
    for target_rank in range(49, 40, -1):
        if time.time() > deadline - 5:
            break
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 5:
                break
            
            # Random initialization: mostly 0s, 1s, -1s, 0.5s
            # We need U (10, r), V (25, r), W (10, r)
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # For each column, assign random non-zero patterns
            for c in range(target_rank):
                # U column: pick a subset of rows
                u_mask = np.random.choice([0, 1, 1, 1, 0], size=n*m)
                u_vals = np.where(u_mask == 1, np.random.choice([-1, 0.5, 1, 1, -0.5]), 0)
                U[:, c] = u_vals
                
                v_mask = np.random.choice([0, 1, 1, 1, 0], size=m*p)
                v_vals = np.where(v_mask == 1, np.random.choice([-1, 0.5, 1, 1, -0.5]), 0)
                V[:, c] = v_vals
                
                w_mask = np.random.choice([0, 1, 1, 1, 0], size=p*n)
                w_vals = np.where(w_mask == 1, np.random.choice([-1, 0.5, 1, 1, -0.5]), 0)
                W[:, c] = w_vals
            
            # Check
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
                # Convert to exact integers/half-integers
                U_out = np.round(U * 2) / 2
                V_out = np.round(V * 2) / 2
                W_out = np.round(W * 2) / 2
                # Verify exactness
                U3 = U_out.reshape(n, m, target_rank)
                V3 = V_out.reshape(m, p, target_rank)
                W3 = W_out.reshape(p, n, target_rank)
                T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                if np.all(T == 1):
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try a more structured approach: modify schoolbook
    # Start with schoolbook and try to eliminate one term
    U_sb = _schoolbook()["U"].astype(np.float64)
    V_sb = _schoolbook()["V"].astype(np.float64)
    W_sb = _schoolbook()["W"].astype(np.float64)
    
    # Try removing one column and adjusting others
    for col_to_remove in range(n * m * p):
        if time.time() > deadline - 5:
            break
        r = n * m * p - 1
        # Try to find adjustments to remaining columns to compensate
        # This is a linear system problem - too complex for quick search
        
    return None

def construct():
    """Construct the decomposition. Returns dict with U, V, W."""
    deadline = time.time() + 120
    
    # First try to find a better decomposition
    result = _try_rank_reduction()
    
    if result is not None:
        # Verify one more time
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
