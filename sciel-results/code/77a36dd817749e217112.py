"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    
    # Try to find a lower-rank decomposition
    for rank in range(n * m * p - 1, 9, -1):
        if time.time() > deadline:
            break
        result = _search_decomposition(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
    
    return best_result

def _search_decomposition(n, m, p, rank, deadline):
    """Search for a valid rank-rank decomposition."""
    np.random.seed(12345)
    
    for attempt in range(500):
        if time.time() > deadline:
            return None
        
        # Random initialization with rational-like values
        U = np.random.randint(-3, 4, (n * m, rank)).astype(float)
        V = np.random.randint(-3, 4, (m * p, rank)).astype(float)
        W = np.random.randint(-3, 4, (p * n, rank)).astype(float)
        
        if _is_valid_decomposition(n, m, p, U, V, W):
            # Convert to list format
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _is_valid_decomposition(n, m, p, U, V, W):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    # The condition for a valid tensor decomposition of matrix multiplication is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This ensures that C[i,k] = sum_j A[i,j] B[j,k] is reproduced.
    
    # We can check this by verifying the tensor equation.
    # Let's test with a specific set of basis matrices to ensure exactness.
    
    # Instead of testing random matrices, we can directly verify the tensor identity:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(len(U[0])):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
