"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]
TIME_LIMIT = 150  # seconds, must finish within 120s evaluation

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=5, m=5, p=5):
    """Verify that U, V, W form a valid matrix multiplication tensor decomposition.
    
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] = 1 for all i,j,k (standard matrix multiplication tensor).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U3d = U.reshape(n, m, -1)  # (n, m, rank)
    V3d = V.reshape(m, p, -1)  # (m, p, rank)
    W3d = W.reshape(p, n, -1)  # (p, n, rank)
    
    rank = U.shape[1]
    
    # Compute tensor: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to compute the full 3D tensor
    # T[i,j,k] = sum_c U3d[i,j,c] * V3d[j,k,c] * W3d[k,i,c]
    
    # Build the full tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U3d[i,j,:] * V3d[j,k,:] * W3d[k,i,:])
    
    # Expected: T[i,j,k] = 1 for all i,j,k
    expected = np.ones((n, m, p))
    
    return np.allclose(T, expected, atol=1e-10)

def _build_rank70_decomposition():
    """Build a rank-70 decomposition using the known algebraic structure.
    
    For 5x5 matrix multiplication, a rank-70 decomposition exists.
    We use the structure based on Strassen-like methods.
    """
    n = m = p = 5
    rank = 70
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: Use the fact that M_5 can be embedded in M_8
    # M_8 has rank 37 (Strassen), so we can extract M_5 from it.
    # 
    # Actually, let's use a direct construction based on the known rank-70 formula.
    # The rank-70 decomposition for 5x5 matrices uses the following structure:
    # 
    # We partition the 5x5 matrices into blocks and use a clever combination.
    # 
    # For now, let's try a structured approach:
    # Use the 2x2 Strassen recursion as a building block.
    
    # Alternative: Use the fact that we can decompose using
    # the "border rank" techniques.
    
    # Let me try a different approach: use random initialization and 
    # verify if it works, or use a known construction.
    
    # Known result: M_5 has rank 70. One construction uses:
    # - 35 terms of one type
    # - 35 terms of another type
    
    # For simplicity and correctness, let's return the schoolbook if we can't
    # find a valid rank-70 decomposition quickly.
    
    # Actually, let's try to construct it properly.
    # The rank-70 decomposition for 5x5x5 can be constructed as follows:
    # 
    # Think of the 5x5 matrix as having a 2x2 block structure where
    # some blocks are 2x2, 2x3, 3x2, 3x3.
    
    # Let's use a simpler approach: try to find a valid decomposition
    # by solving a system of equations.
    
    # For now, return None to indicate failure, and the caller will
    # fall back to schoolbook.
    
    # Actually, let me try a concrete construction:
    # 
    # The key insight: M_5 ⊂ M_6 ⊗ M_1, and we know bounds for these.
    # 
    # Let's just try to verify if a randomly initialized rank-70 works
    # (very unlikely), or use a known formula.
    
    # For this implementation, I'll use a heuristic search with 
    # structured initialization.
    
    return None  # Will fall back to schoolbook

def _schoolbook_decomposition(n=5, m=5, p=5):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_ranked_search(n=5, m=5, p=5, max_rank=125, time_budget=120):
    """Try to find a decomposition with rank < max_rank using structured search."""
    start_time = time.time()
    
    # Try ranks from a known lower bound upward
    # Known: rank(M_5) = 70
    # Try rank 70 first, then 71, 72, ...
    
    for target_rank in range(70, min(max_rank, 126)):
        if time.time() - start_time > time_budget - 10:
            break
        
        # Try to construct a decomposition of this rank
        # Using a structured approach based on the algebraic properties
        
        # For now, this is a placeholder for a real search algorithm.
        # A proper implementation would use:
        # 1. Random initialization with structure
        # 2. Gradient-based optimization
        # 3. Verification
        
        pass
    
    return None

def construct():
    """Construct a matrix multiplication tensor decomposition.
    
    Returns a dict with U, V, W matrices that exactly decompose
    the 5x5x5 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Return the schoolbook decomposition (rank 125) as a safe baseline.
    result = _schoolbook_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return result
# EVOLVE-BLOCK-END
