"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x8 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (24, rank)  
    - W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies: for all i,k,j:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} pattern
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Schoolbook baseline: rank = n*m*p = 72
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured search
    # Key insight: for matrix multiplication, we can use block-structured approaches
    
    # Try Strassen-like decomposition for the 3x3 * 3x8 case
    # We can view this as multiplying 3x3 by 3x8 = 3x3 * [I, 0] style blocks
    
    # Approach: Use the fact that 3x8 = 3x3 * diag(1,1,1) extended
    # Try rank values from low to high
    
    def validate(U_list, V_list, W_list):
        """Validate that the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give (AB)_{i,k}
        # This means: sum_c U[i*m+j,c]*V[j*p+k,c] should equal delta pattern
        # More precisely: the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # should equal 1 if this corresponds to (AB)_{i,k} contribution from B_{j,k}
        
        # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # because (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    def try_structured_rank(r, max_iters=1000):
        """Try to find a structured rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        # For matrix multiplication of (n x m) * (m x p), a rank-r decomposition
        # corresponds to r pairs of (n x m) and (m x p) matrices whose outer products
        # sum to the multiplication tensor
        
        # Try random initialization with structured patterns
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Try gradient descent to minimize the error
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] * (sum - 1)
                            # d/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c] * (sum - 1)
                            # d/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c] * (sum - 1)
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err = s - 1.0
                            error += err * err
                            for c in range(r):
                                grad_U[i * m + j, c] += V[j * p + k, c] * W[k * n + i, c] * err
                                grad_V[j * p + k, c] += U[i * m + j, c] * W[k * n + i, c] * err
                                grad_W[k * n + i, c] += U[i * m + j, c] * V[j * p + k, c] * err
                
                if error < 1e-15:
                    break
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if step % 10 == 0:
                    lr *= 0.5
            
            if error < 1e-10:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if validate(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks from known lower bounds up to schoolbook
    # For 3x3 * 3x8, the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actually for matrix multiplication, rank >= n*m*p / something... 
    # Let's try from reasonable lower bound
    
    # Known: 3x3*3x3 has rank 21 (Strassen). For 3x3*3x8, we can extend.
    # Try ranks: 24, 25, 26, ... up to 72
    
    for r in range(24, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_structured_rank(r, max_iters=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
