"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] = 1."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _decompose_to_matrices(U, V, W, n, m, p):
    """Convert U, V, W factor matrices to standard form."""
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_decomp = np.zeros((n, m, p), dtype=np.int64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    return np.array_equal(T, T_decomp)

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    # For small ranks, we can try systematic search
    # The tensor has n*m*p = 18 nonzero entries
    # Each rank-1 term can cover at most one (i,j,k) entry
    # So we need at least 18 terms... but wait, a single rank-1 term
    # U[i,j,c] * V[j,k,c] * W[k,i,c] can be nonzero for multiple (i,j,k)
    # if the same c is used with different i,j,k values.
    
    # Actually, for a rank-1 term with fixed c:
    # U[i*m+j, c] is a value, V[j*p+k, c] is a value, W[k*n+i, c] is a value
    # The product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] contributes to T[i,j,k]
    # For this to be nonzero for multiple (i,j,k), we need:
    # - U[i*m+j,c] nonzero for multiple (i,j)
    # - V[j*p+k,c] nonzero for multiple (j,k)  
    # - W[k*n+i,c] nonzero for multiple (k,i)
    
    # For the all-ones tensor, each entry T[i,j,k]=1
    # A rank-1 term with U[i,j,c]=a, V[j,k,c]=b, W[k,i,c]=c_val
    # contributes a*b*c_val to T[i,j,k]
    # For this to equal 1 for all (i,j,k), we'd need a*b*c_val=1
    # But a is fixed for given (i,j), so this can't work for multiple entries
    # unless we're careful.
    
    # Actually, the schoolbook uses rank 18 with each term covering exactly one entry.
    # Can we do better? Let's think about it differently.
    
    # For 2x3 x 3x3, the known lower bounds and upper bounds:
    # The schoolbook gives 18.
    # There might be optimizations, but for this small case, let's try a search.
    
    # Strategy: try to find a decomposition with rank r < 18
    # We'll use a backtracking search on the tensor entries.
    
    # For now, let's just verify that rank 18 works (schoolbook) and try
    # to find if rank 17 or lower works.
    
    # Given the time constraints, let's do a limited search.
    # The key insight: for the all-ones tensor, we can potentially combine
    # some terms.
    
    # Let's try a simple approach: start with schoolbook and try to merge
    # or eliminate terms.
    
    # Actually, for this specific small case, let me just return the schoolbook
    # since a full search would be too expensive. But let me try a few
    # specific patterns.
    
    if r >= n * m * p:
        # Schoolbook works
        U = np.zeros((n*m, r), dtype=np.int64)
        V = np.zeros((m*p, r), dtype=np.int64)
        W = np.zeros((p*n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        # Only use first r columns if r < n*m*p, but we need all n*m*p
        # So this only works if r == n*m*p
        if r == n * m * p:
            return U[:n*m, :r], V[:m*p, :r], W[:p*n, :r]
        return None
    
    # For r < n*m*p, we need a smarter decomposition
    # This is a research problem; for now, return None to indicate failure
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find better decompositions
    # For 2x3 x 3x3, try ranks from 17 down
    for r in range(rank - 1, 0, -1):
        if time.time() > deadline - 30:  # Safety margin
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U_arr, V_arr, W_arr = result
            if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
                best_rank = r
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                break  # Found the best, stop
    
    return best_result
# EVOLVE-BLOCK-END
