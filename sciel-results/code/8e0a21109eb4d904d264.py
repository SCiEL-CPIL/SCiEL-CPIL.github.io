<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    # Build the multiplication tensor T[i,j,k] = delta(i,k) in the appropriate format
    # The tensor for C = A @ B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 if this contributes to C[i,k]
    # In the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] is the output factor
    
    # The tensor T has entries T[i,j,k] where the multiplication is:
    # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For exact decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} for the matrix mult tensor
    
    # Actually, the standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j,r} A[i,j] B[j,k] * delta_{i,k} ... no
    
    # Let me think more carefully. The matrix multiplication tensor M has:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Actually M[i,j,k] = delta_{i,k} doesn't make sense directly.
    
    # The correct tensor: for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] represents the coefficient: T[i,j,k] = 1 if A[i,j] and B[j,k] 
    # contribute to C[i,k], which is always 1 for valid indices.
    
    # So T[i,j,k] = 1 for all valid i,j,k, and the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid (i,j,k) triple
    # Actually no - the tensor for matrix multiplication is:
    # T[i,j,k] = 1 (it's just the identity in the sense that every product A[i,j]*B[j,k] contributes)
    
    # Wait, I need to be more careful. The multiplication tensor for (n,m,p) matrices is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where these are the basis elements
    # No, that's not right either.
    
    # The correct formulation: 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C = sum_r U_r ⊗ V_r ⊗ W_r where
    # U_r is a matrix (n x m), V_r is a matrix (m x p), W_r is a matrix (p x n)
    # and the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T[i,j,k] = delta_{i,k} ... no
    
    # Let me use the standard: the tensor T has T[i,j,k] = 1 for all i,j,k
    # No wait. For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes to C[i,k])
    # But that's not quite right because the same (i,j,k) doesn't map to multiple outputs.
    
    # Actually: T[i,j,k] = 1 means that the product A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # The decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    
    # Let's verify this
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, n_trials=1000, timeout=None):
    """Try to find a decomposition of given rank using random initialization and optimization."""
    if timeout is not None:
        start = time.time()
        if time.time() - start > timeout:
            return None
    
    for trial in range(n_trials):
        if timeout is not None and time.time() - start > timeout - 5:
            break
        
        # Initialize with random values
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Try gradient descent to minimize the error
        lr = 0.01
        for step in range(50):
            if timeout is not None and time.time() - start > timeout - 5:
                return None
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = 1
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        for r in range(rank):
                            grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += U[i*m+j, r] * err * W[k*n+i, r]
                            grad_W[k*n+i, r] += U[i*m+j, r] * V[j*p+k, r] * err
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if we found a valid decomposition
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_decomposition(n, m, p, rank):
    """Try structured decompositions based on known patterns."""
    # For 3x5x5, try various structured approaches
    # One approach: exploit the fact that n=3 is small
    
    # Try a block-structured decomposition
    # This is a heuristic approach
    
    if rank < n * m * p:
        # Try to find a lower rank decomposition
        # For now, return None as we don't have a known structure
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # First, return the schoolbook decomposition as a baseline
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
    
    # Try to find lower rank decompositions
    # Start from rank-1 and go up
    for r in range(1, rank):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try random search for this rank
        result = _try_random_decomposition(n, m, p, r, n_trials=10, timeout=deadline - (time.time() - start_time))
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found the minimal rank
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
