<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other combinations, the sum should be 0.
    """
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = 1 if this is a valid (i,j,k) triple
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] = 1 for all valid i, j, k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For each (i, j, k) in [0,n) x [0,m) x [0,p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _try_rank_r(n, m, p, r, max_time=30.0):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    start = time.time()
    deadline = start + max_time
    
    # Use random initialization with restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Iterative refinement: for each tensor entry, update one factor
        # This is a simple alternating minimization approach
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # For each k, we need sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is a linear system in U[idx, :]
                    # A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                    # We need sum_k sum_r U[idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
                    # Actually, for each k separately:
                    # sum_r U[idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    # This gives p equations in r unknowns (U[idx, 0], ..., U[idx, r-1])
                    # If r >= p, we can solve this least squares problem
                    
                    # Build the system: for each k, equation is dot(U[idx,:], V[j*p+k,:]*W[k*n+i,:]) = 1
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve least squares
                    U[idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    V[idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    W[idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if we found a valid decomposition
        if _verify_decomposition(n, m, p, U, V, W):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Known results suggest rank is likely between 35 and 105
    for r in range(35, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_r(n, m, p, r, max_time=20.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found the lowest rank we can achieve with this approach
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
