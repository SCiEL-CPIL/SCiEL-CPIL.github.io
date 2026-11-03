"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _validate_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    
    # Check all valid tensor entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that invalid entries are 0
    for a in range(n * m):
        i1 = a // m
        j1 = a % m
        for b in range(m * p):
            j2 = b // p
            k1 = b % p
            if j1 != j2:
                continue
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                if i1 != i2 or k1 != k2:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W

def _try_structured_search(n, m, p, deadline):
    """
    Search for lower-rank decompositions using structured approaches.
    Returns (U, V, W) if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Strategy: Try to find a decomposition by solving a linear system
    # For rank r, we have r*(nm+mp+pn) unknowns and nm*mp*pn equations
    # But the tensor is sparse, so we only need to satisfy the nonzero entries
    
    # Let's try a randomized approach:
    # 1. Initialize with random vectors
    # 2. Use gradient descent or iterative projection to satisfy constraints
    
    np.random.seed(42)
    
    for target_rank in range(119, 80, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Use iterative method: for each target entry, adjust the vectors
            # to make the product closer to 1
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current values for all target entries
                max_error = 0
                for (a, b, c) in target_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error = 1.0 - current_val
                    max_error = max(max_error, abs(error))
                
                if max_error < 1e-6:
                    break
                
                # Gradient step: for each (a,b,c), the gradient w.r.t. U[a,:] is V[b,:]*W[c,:]
                # Update: U[a,:] += learning_rate * error * V[b,:]*W[c,:]
                lr = 0.01
                for (a, b, c) in target_entries:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error = 1.0 - current_val
                    
                    # Gradient for U[a,:]
                    grad_U = V[b, :] * W[c, :]
                    U[a, :] += lr * error * grad_U
                    
                    # Gradient for V[b,:]
                    grad_V = U[a, :] * W[c, :]
                    V[b, :] += lr * error * grad_V
                    
                    # Gradient for W[c,:]
                    grad_W = U[a, :] * V[b, :]
                    W[c, :] += lr * error * grad_W
                
                # Also need to ensure invalid entries are 0
                # This is harder - we need to add constraints
                # For now, let's just check if we're making progress
    
    return None

def _try_block_decomposition(n, m, p, deadline):
    """
    Try a block-based approach to find lower rank.
    Idea: group indices and use shared rank terms.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # For 4x5x6, let's try to exploit the structure
    # Group the j-index (size 5) and see if we can share terms
    
    # Approach: use a "factorization" where we split the sum over j
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we can write this as a sum of rank-1 terms in a clever way...
    
    # Let's try: for each pair (i,k), we need sum_j A[i,j]B[j,k]
    # This is inherently a sum of 5 terms. Can we reduce this?
    
    # One idea: if some rows of A or columns of B are linearly dependent,
    # we can reduce the rank. But for generic matrices, we can't.
    
    # However, for the tensor decomposition, we're looking for a fixed
    # set of rank terms that work for ALL matrices, not just specific ones.
    
    # Let's try a different approach: use the known result that for
    # m x n multiplication, the rank is at most mn + m + n - 1 (for some cases)
    # But this is for square matrices.
    
    # For now, let's just try to verify if rank 119 or 118 is possible
    # by checking if the schoolbook decomposition has linear dependencies
    
    U, V, W = _schoolbook_decomposition(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if any column is a linear combination of others
    # This would indicate we can reduce the rank
    # But for the schoolbook, each column corresponds to a unique (i,j,k),
    # so they should be linearly independent
    
    return None

def construct():
    """
    Returns the best matrix multiplication tensor decomposition found.
    Starts with schoolbook and tries to find lower-rank alternatives.
    """
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    result = _try_structured_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        best_rank = U.shape[1]
    
    result2 = _try_block_decomposition(n, m, p, deadline)
    if result2 is not None:
        U2, V2, W2 = result2
        if U2.shape[1] < best_rank:
            U, V, W = U2, V2, W2
            best_rank = U2.shape[1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
