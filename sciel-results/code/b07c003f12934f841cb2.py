import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find rank R and matrices U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use the fact that for matrix multiplication, we can use
    # the Strassen-like approach or direct structured decomposition.
    
    # For 2x4 * 4x7, let's try to find a decomposition.
    # The schoolbook rank is 2*4*7 = 56.
    # Known results suggest rank can be lower.
    
    # Let's use a structured approach based on the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as a sum of rank-1 terms.
    # Each term r contributes: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # A simple structured approach:
    # For each (i, k), we need sum_j A[i,j] B[j,k]
    # This is like an inner product over j.
    
    # Let's try to use the identity:
    # For each j, we have a term that picks out A[i,j] and B[j,k]
    # This gives rank m = 4 terms? No, that's not right.
    
    # Actually, the standard decomposition uses rank n*m*p terms where each term
    # corresponds to a single (i,j,k) triple.
    
    # Let's try a more clever approach. We can group terms.
    
    # For fixed i and k, C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as a sum over j of (A[i,j] * B[j,k])
    # Each term A[i,j] * B[j,k] can be represented as a rank-1 tensor
    # where U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # But this gives rank n*m*p = 56.
    
    # To reduce rank, we need to combine terms.
    
    # Let's try a random search with alternating minimization.
    
    def verify(U, V, W, R):
        """Verify that the decomposition is exact."""
        # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def alternating_minimization(R, max_iters=200, tol=1e-12):
        """Try to find a decomposition of rank R using alternating minimization."""
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U: fix V, W
            # For each (i, j), we want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is not linear in U alone.
            
            # Alternative: Use the fact that for each r, we have a rank-1 tensor.
            # The sum should equal T.
            
            # Let's use a different approach: project onto the constraint.
            
            # For each (i, j, k), the constraint is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # We can try to minimize the residual.
            
            # Compute current residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        residual += (val - 1.0) ** 2
            
            if residual < tol:
                return U.copy(), V.copy(), W.copy()
            
            # Update each factor
            # For U: fix V, W. For each (i, j), we want to find U[i*m+j, :]
            # such that sum_k (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) = 1
            # This is: sum_k W[k*n+i, :]^T * (U[i*m+j, :] * V[j*p+k, :]) = 1
            # Let A_k = W[k*n+i, :] * V[j*p+k, :] (element-wise product)
            # Then sum_k A_k^T * U[i*m+j, :] = 1
            # This is: (sum_k A_k)^T * U[i*m+j, :] = 1
            # So U[i*m+j, :] should be in the direction of sum_k A_k
            
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Compute sum_k W[k*n+i, :] * V[j*p+k, :]
                    s = np.zeros(R)
                    for k in range(p):
                        s += W[k*n+i, :] * V[j*p+k, :]
                    if np.linalg.norm(s) > 1e-15:
                        # We want s^T * U[i*m+j, :] = 1
                        # Minimum norm solution: U[i*m+j, :] = s / (s^T * s)
                        new_U[i*m+j, :] = s / (s @ s)
                    else:
                        new_U[i*m+j, :] = U[i*m+j, :]
            U = new_U
            
            # Similarly for V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    s = np.zeros(R)
                    for i in range(n):
                        s += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(s) > 1e-15:
                        new_V[j*p+k, :] = s / (s @ s)
                    else:
                        new_V[j*p+k, :] = V[j*p+k, :]
            V = new_V
            
            # Similarly for W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    s = np.zeros(R)
                    for j in range(m):
                        s += U[i*m+j, :] * V[j*p+k, :]
                    if np.linalg.norm(s) > 1e-15:
                        new_W[k*n+i, :] = s / (s @ s)
                    else:
                        new_W[k*n+i, :] = W[k*n+i, :]
            W = new_W
        
        # Check if we converged
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (val - 1.0) ** 2
        
        if residual < 1e-10:
            return U.copy(), V.copy(), W.copy()
        return None
    
    # Try ranks from low to high
    best_result = None
    for R in range(15, 35):
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 12345 + R)
            result = alternating_minimization(R)
            if result is not None:
                U, V, W = result
                if verify(U.tolist(), V.tolist(), W.tolist(), R):
                    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition
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
