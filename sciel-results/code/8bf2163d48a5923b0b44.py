"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i==k (for C[i,k] = sum_j A[i,j]*B[j,k]) and 0 otherwise
    # Actually for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if this term contributes, which is always 1 for valid indices
    # Wait, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me reconsider. The matrix multiplication tensor M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_like_3x3x6(n, m, p):
    """
    Try to find a lower-rank decomposition by exploiting structure.
    
    For 3x3x6, we can think of it as two 3x3x3 multiplications.
    If we can find rank r for 3x3x3, we might do better than 27*2=54.
    
    One approach: use the fact that we're multiplying by a 3x6 matrix.
    We can write B as [B1 | B2] where B1 and B2 are 3x3.
    Then C = [A*B1 | A*B2].
    
    If we use the same U factors for both halves, we can share computation.
    """
    # Try rank 40-53 by searching for structured decompositions
    # For now, try a few specific ranks
    
    best_result = None
    
    # Try to construct a decomposition with rank < 54
    # One idea: use block structure
    
    # Let's try a systematic search for small modifications to schoolbook
    # that might reduce rank
    
    # Actually, let's try to use the structure more directly.
    # For 3x3x6, we can view it as computing C = A @ B where B is 3x6.
    # B = [B_left | B_right] where each is 3x3.
    # C = [A @ B_left | A @ B_right]
    
    # If we precompute some linear combinations of rows of A, we might save.
    
    # Try rank 45, 48, 50, 52, 53
    for target_rank in [53, 52, 51, 50, 48, 45, 40, 36]:
        # Simple heuristic: try to merge some columns
        # This is a placeholder - real search would be more sophisticated
        
        # For now, just return schoolbook if we can't find better
        pass
    
    return None

def _random_search(n, m, p, max_iter=1000, time_limit=10):
    """Random search for low-rank decompositions."""
    start_time = time.time()
    best_rank = n * m * p
    best_result = None
    
    # Try random initializations and local optimization
    for target_rank in range(30, min(54, n*m*p + 1)):
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                return best_result
            
            # Random initialization with small entries
            np.random.seed(42 + target_rank * 1000 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple local optimization: gradient descent on the error
            # Error = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            lr = 0.01
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error += (val - 1.0) ** 2
                
                if error < 1e-10:
                    # Found a valid decomposition!
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    return best_result
                
                # Gradient computation (simplified - this is expensive)
                # For now, just do random perturbations
                U += np.random.randn(*U.shape) * lr * 0.1
                V += np.random.randn(*V.shape) * lr * 0.1
                W += np.random.randn(*W.shape) * lr * 0.1
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Deadline guard: 150s wall-clock, use 140s as safety margin
    deadline = time.time() + 140
    
    # Try random search with time budget
    random_start = time.time()
    random_result = _random_search(n, m, p, max_iter=500, time_limit=60)
    if random_result is not None:
        # Verify
        if _verify_decomposition(random_result["U"], random_result["V"], random_result["W"], n, m, p):
            result = random_result
            best_rank = len(random_result["U"][0])
    
    # Try structured approaches if time permits
    if time.time() < deadline - 30:
        # Try more systematic search
        pass
    
    return result
# EVOLVE-BLOCK-END
