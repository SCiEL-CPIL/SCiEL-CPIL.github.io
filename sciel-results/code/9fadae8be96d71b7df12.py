"""Matrix multiplication 2x3x7 tensor decomposition - distinct implementation."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition using numpy kron-based construction."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Check: for each rank term c, U[:,c] * V[:,c] * W[:,c] should sum to the tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The correct tensor has T[i,j,k] = 1 if and only if the multiplication is well-defined
    # Actually the tensor for matrix mult is: C = A @ B where A is n x m, B is m x p, C is n x p
    # T[i,j,k] should equal 1 for all (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce(U, V, W, n, m, p, max_time=120):
    """Try to reduce rank by finding redundant columns."""
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Try to find columns that are linear combinations of others
    # This is a heuristic: try removing one column at a time and see if we can
    # adjust the remaining columns to compensate
    
    # A simpler approach: try Gaussian elimination style - find if any column
    # is in the span of others for each of U, V, W simultaneously
    # This is complex, so let's try a random approach
    
    rng = np.random.default_rng(42)
    improved = False
    
    # Try small random perturbations and rank reduction
    # This is very unlikely to succeed quickly, but let's try
    for attempt in range(100):
        if time.time() - start > max_time - 5:
            break
        # Try to zero out a random column by adjusting others
        col_to_remove = rng.integers(0, rank)
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        
        # Check if column col_to_remove is "redundant" - i.e., its contribution
        # can be absorbed by other columns
        # For this to work, we need U[:,col] to be in the span of other U columns,
        # V[:,col] in span of other V columns, and W[:,col] in span of other W columns
        # simultaneously, which is very restrictive
        
        # Quick check: if the column is all zeros, remove it
        if np.allclose(U[:, col_to_remove], 0) and np.allclose(V[:, col_to_remove], 0) and np.allclose(W[:, col_to_remove], 0):
            U = np.delete(U, col_to_remove, axis=1)
            V = np.delete(V, col_to_remove, axis=1)
            W = np.delete(W, col_to_remove, axis=1)
            rank -= 1
            improved = True
            break
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify the schoolbook is correct
    assert _verify(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    # Try to reduce (unlikely to succeed quickly)
    U, V, W, rank = _try_reduce(U, V, W, n, m, p, max_time=110)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        rank = n * m * p
    
    # Convert to lists of integers/half-integers
    U_list = [[int(round(U[i, c] * 2)) / 2 for c in range(rank)] for i in range(n * m)]
    V_list = [[int(round(V[i, c] * 2)) / 2 for c in range(rank)] for i in range(m * p)]
    W_list = [[int(round(W[i, c] * 2)) / 2 for c in range(rank)] for i in range(p * n)]
    
    # Clean up: replace 0.0 with 0, 1.0 with 1
    def clean(val):
        v = int(round(val * 2)) / 2
        if v == 0:
            return 0
        elif v == 1:
            return 1
        return v
    
    U_list = [[clean(U_list[i][c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[clean(V_list[i][c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[clean(W_list[i][c]) for c in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
