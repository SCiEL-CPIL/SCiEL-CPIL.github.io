"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 5 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 145  # 5s safety margin before 150s limit
    
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is valid
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # T[i,j,k] = 1 if j is the shared index, else 0
    # More precisely: T[i,j,k] = 1 if the multiplication term A[i,j]*B[j,k] contributes to C[i,k]
    
    # The schoolbook decomposition has rank n*m*p = 2*3*5 = 30
    # But we can do better.
    
    # Known results: For 2x3 * 3x5 matrix multiplication
    # The rank is at most min(n*m*p, some bound)
    # Lower bound: max(n*m, m*p, n*p) = max(6, 15, 10) = 15
    # Upper bound: n*m*p = 30
    
    # Let's try to find a decomposition with rank < 30
    
    # Strategy: Use the fact that we can combine some terms
    # For example, if we have two terms that share the same (i,k) pair but different j,
    # we might be able to combine them if the U and W entries allow it.
    
    # Actually, let's think about this more carefully.
    # The tensor T is defined by: T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] exists
    # In standard matrix multiplication, ALL terms exist, so T[i,j,k] = 1 for all valid (i,j,k)
    
    # Wait, I need to re-read the problem. The tensor decomposition is for the matrix multiplication
    # operation. The tensor has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k), and we want:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # The schoolbook solution uses one term for each (i,j,k) triple, giving rank n*m*p = 30.
    
    # Can we do better? Yes, if we can find a lower-rank decomposition.
    
    # Let's try rank 15 (the lower bound) or something in between.
    
    # Approach: Try to find a decomposition with rank r < 30
    # We'll use a randomized approach with local optimization
    
    def make_schoolbook():
        """Create the schoolbook decomposition of rank 30."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        rank = U.shape[1]
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    # The tensor entry should be 1
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        return max_err < tol, max_err
    
    def round_to_half_int(x, tol=1e-6):
        """Round to nearest half-integer if close."""
        r = round(2 * x) / 2
        if abs(x - r) < tol:
            return r
        return x
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] should be 1
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err = val - 1.0
                        for c in range(r):
                            dU[i * m + j, c] += err * V[j * p + k, c] * W[k * n + i, c]
                            dV[j * p + k, c] += err * U[i * m + j, c] * W[k * n + i, c]
                            dW[k * n + i, c] += err * U[i * m + j, c] * V[j * p + k, c]
            
            # Gradient descent step
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check error
            err, _ = check_decomposition(U, V, W, tol=1e-4)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-4:
            # Try to round to half-integers
            U_rounded = np.vectorize(lambda x: round_to_half_int(x))(best_U)
            V_rounded = np.vectorize(lambda x: round_to_half_int(x))(best_V)
            W_rounded = np.vectorize(lambda x: round_to_half_int(x))(best_W)
            
            err_rounded, _ = check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6)
            if err_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
            else:
                # Return the unrounded version if it's close enough
                if best_err < 1e-5:
                    return best_U, best_V, best_W
        
        return None
    
    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school = make_schoolbook()
    
    # Try to find a better solution
    # Start with rank 15 (lower bound) and increase if needed
    for r in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=2000, seed=42 + r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
