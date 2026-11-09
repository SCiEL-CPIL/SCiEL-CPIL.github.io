"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a sparse representation."""
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries.append((a, b, c, 1.0))
    return entries

def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    entries = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Check each non-zero entry
    for a, b, c, val in entries:
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s - val) > tol:
            return False
    # Check that all other entries are zero (sample some)
    # For efficiency, check a subset
    checked = set((a, b, c) for a, b, c, _ in entries)
    # Check some random zero entries
    for _ in range(200):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        if (a, b, c) in checked:
            continue
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    return True

def _try_eliminate_term(n, m, p, U, V, W, target_col, time_limit):
    """
    Try to eliminate the term at target_col by adjusting other terms.
    Returns new (U, V, W) with one fewer column if successful, else None.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # We want to set column target_col to zero and adjust others
    # This is a nonlinear problem. Use a simpler approach:
    # Try to express the target term as a linear combination of others
    
    # Alternative: use gradient-based optimization to minimize the norm
    # of the target column while keeping the decomposition valid
    
    start_time = time.time()
    
    # Create a copy
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    # We'll try to zero out column target_col by adjusting other columns
    # This is equivalent to finding if the target rank-1 tensor is in the
    # span of the other rank-1 tensors
    
    # Use a simple approach: try random perturbations of other columns
    # to see if we can absorb the target term
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    # Number of attempts based on time
    max_attempts = 50
    per_attempt_time = time_limit / max_attempts
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization around current solution
        scale = 0.1
        U_try = U_work.copy()
        V_try = V_work.copy()
        W_try = W_work.copy()
        
        # Zero out target column
        U_try[:, target_col] = 0
        V_try[:, target_col] = 0
        W_try[:, target_col] = 0
        
        # Add random perturbations to other columns
        for r in range(rank):
            if r == target_col:
                continue
            U_try[:, r] += np.random.randn(n * m) * scale
            V_try[:, r] += np.random.randn(m * p) * scale
            W_try[:, r] += np.random.randn(p * n) * scale
        
        # Verify
        if _verify_decomposition(n, m, p, U_try, V_try, W_try, tol=1e-6):
            # Try to polish
            # Remove the zero column
            mask = np.ones(rank, dtype=bool)
            mask[target_col] = False
            U_new = U_try[:, mask]
            V_new = V_try[:, mask]
            W_new = W_try[:, mask]
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    time_start = time.time()
    deadline = 130  # Safety margin from 150s limit
    
    # Build schoolbook decomposition
    rank = n * m * p  # 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    assert _verify_decomposition(n, m, p, U, V, W), "Schoolbook decomposition failed"
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to eliminate terms one by one
    # Order: try to eliminate terms that might be most redundant
    # Terms with same j are in the same "slice"
    
    # Create a list of columns to try eliminating
    # Group by j: for each j, we have n*p terms
    # Try eliminating within each j-group first
    
    col_order = []
    for j in range(m):
        for i in range(n):
            for k in range(p):
                col_order.append(i * m + j * p + k)  # This is wrong, let me fix
    
    # Actually, the column index is just sequential: col = i*m*p + j*p + k? No.
    # From the loop: col increments as i, j, k all range
    # So col for (i,j,k) = i*m*p + j*p + k? Let's check:
    # i=0,j=0,k=0: col=0
    # i=0,j=0,k=1: col=1
    # ...
    # i=0,j=0,k=7: col=7
    # i=0,j=1,k=0: col=8
    # So col = i*(m*p) + j*p + k
    
    # Let's try eliminating in a specific order
    # Start with terms that have the same (i,j) but different k, or same (j,k) but different i
    
    # Try each column, see if we can eliminate it
    for target in range(rank):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
        
        # Try to eliminate this column
        result = _try_eliminate_term(n, m, p, best_U, best_V, best_W, target, 
                                      min(remaining_time * 0.5, 5.0))
        if result is not None:
            best_U, best_V, best_W = result
            best_rank -= 1
            # After successful elimination, the column indices shift
            # We need to adjust: the removed column was at 'target', 
            # so subsequent columns shift left
            # For simplicity, restart the elimination process with new matrices
            break  # Restart with updated matrices
    
    # Convert to lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
