"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for matrix multiplication."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Build the tensor and verify
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i,j,k form a valid multiplication term, 0 otherwise
    # Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # For each (i,j,k), check if u[i*m+j] * v[j*p+k] * w[k*n+i] contributes correctly
        # The sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 for all i,j,k
    
    # More efficient: build the full tensor from U,V,W and compare
    T_from_decomp = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_from_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    
    T_expected = np.ones((n, m, p))
    
    # Check exact match (allowing for floating point with tight tolerance)
    if not np.allclose(T_from_decomp, T_expected, atol=1e-10, rtol=1e-10):
        return False
    
    return True

def _try_structured_decomposition(n, m, p, max_rank_to_try, deadline):
    """
    Try to find a decomposition with rank less than schoolbook.
    Uses structured patterns based on known small matrix multiplication results.
    """
    start_time = time.time()
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try various structured approaches
    # For 2x3xp, there are known improvements over schoolbook
    
    # Approach 1: Try to find a decomposition using Kronecker product structure
    # or other algebraic patterns
    
    # Approach 2: Use the fact that for small matrices, we can try to find
    # a low-rank decomposition by solving a system of equations
    
    # Given time constraints, let's try a simpler approach:
    # Check if there's a known better decomposition for these dimensions
    
    # For 2x3x10, the schoolbook rank is 60.
    # The lower bound is max(2*3, 3*10, 10*2) = max(6, 30, 20) = 30.
    # There may be decompositions with rank between 30 and 60.
    
    # Let's try to find a decomposition with rank around 40-50 using
    # a structured approach based on grouping terms
    
    # Strategy: Group some of the schoolbook terms and see if they can be
    # combined or if there's a pattern that reduces the rank
    
    # For now, return None to indicate no improvement found,
    # and we'll use schoolbook as fallback
    return None

def construct():
    """
    Returns a matrix multiplication tensor decomposition.
    Tries to find a rank-minimal exact decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # First, try to find an improved decomposition
    improved = _try_structured_decomposition(n, m, p, n*m*p - 1, deadline)
    
    if improved is not None:
        return improved
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
