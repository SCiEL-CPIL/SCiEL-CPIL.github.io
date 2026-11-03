"""Matrix multiplication 3,4,7 - low-rank tensor decomposition."""
import time
import sys
from fractions import Fraction
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # leave 10s margin before 150s limit
    
    # The tensor decomposition condition:
    # For matrix multiplication C = A @ B where C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Strategy: Use gradient-based optimization with multiple restarts
    # and a smart initialization based on block structure.
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return np.sqrt(err)
    
    def compute_error_vectorized(U, V, W):
        """Vectorized error computation."""
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for each j
                for j in range(m):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        return np.sqrt(err)
    
    def compute_error_fast(U, V, W):
        """Fast error computation using einsum."""
        r = U.shape[1]
        total_err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    total_err += (s - 1.0) ** 2
        return np.sqrt(total_err)
    
    def gradient_descent(r, max_iter=200, lr_init=0.01, lr_decay=0.999):
        """Perform gradient descent with adaptive line search to find a rank-r decomposition."""
        if r < 1:
            return None
        
        # Random initialization with structured components
        np.random.seed(42 + r)
        
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Normalize columns
        for c in range(r):
            nu = norm(U[:, c])
            nv = norm(V[:, c])
            nw = norm(W[:, c])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                U[:, c] /= nu
                V[:, c] /= nv
                W[:, c] /= nw
        
        lr = lr_init
        best_err = compute_error_fast(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        stall_count = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        # Gradient of (s-1)^2 with respect to each factor
                        factor = 2.0 * (s - 1.0)
                        dU[i*m+j, :] += factor * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += factor * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += factor * U[i*m+j, :] * V[j*p+k, :]
            
            # Adaptive line search
            best_step_err = best_err
            best_step_U, best_step_V, best_step_W = U.copy(), V.copy(), W.copy()
            best_alpha = lr
            
            for alpha in [lr, lr * 0.5, lr * 2.0, lr * 0.25, lr * 4.0]:
                test_U = U - alpha * dU
                test_V = V - alpha * dV
                test_W = W - alpha * dW
                test_err = compute_error_fast(test_U, test_V, test_W)
                if test_err < best_step_err:
                    best_step_err = test_err
                    best_step_U, best_step_V, best_step_W = test_U, test_V, test_W
                    best_alpha = alpha
            
            U, V, W = best_step_U, best_step_V, best_step_W
            lr = best_alpha * lr_decay
            
            # Decay learning rate
            lr = max(lr, 1e-6)
            
            # Compute error
            err = best_step_err
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if err < 1e-8:
                break
            
            if stall_count > 20:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None
    
    def try_structured_init(r):
        """Try a structured initialization based on block decomposition."""
        # For 3x4x7, use a smarter initialization
        np.random.seed(123 + r)
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Strategy: Initialize with a partial schoolbook-like structure
        # then add random components to allow for rank reduction
        
        # First, set up a base structure that covers most entries
        # Use a cyclic pattern to spread out the contributions
        for c in range(min(r, n*m*p)):
            # Map c to (i,j,k) with a different ordering
            k = c % p
            j = (c // p) % m
            i = (c // (p*m)) % n
            
            if c < r:
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
        
        # Add structured perturbations that preserve some symmetry
        for c in range(r):
            # Add small random values to non-zero entries
            mask_u = U[:, c] != 0
            mask_v = V[:, c] != 0
            mask_w = W[:, c] != 0
            U[mask_u, c] += np.random.randn(mask_u.sum()) * 0.3
            V[mask_v, c] += np.random.randn(mask_v.sum()) * 0.3
            W[mask_w, c] += np.random.randn(mask_w.sum()) * 0.3
        
        # Normalize
        for c in range(r):
            nu = norm(U[:, c])
            nv = norm(V[:, c])
            nw = norm(W[:, c])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                U[:, c] /= nu
                V[:, c] /= nv
                W[:, c] /= nw
        
        return U, V, W
    
    def polish(U, V, W, max_iter=50, tol=1e-10):
        """High-precision polish pass with adaptive step size."""
        r = U.shape[1]
        lr = 0.001
        
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        factor = 2.0 * (s - 1.0)
                        dU[i*m+j, :] += factor * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += factor * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += factor * U[i*m+j, :] * V[j*p+k, :]
            
            # Adaptive line search for polishing
            best_step_err = compute_error_fast(U, V, W)
            best_step_U, best_step_V, best_step_W = U.copy(), V.copy(), W.copy()
            best_alpha = lr
            
            for alpha in [lr, lr * 0.5, lr * 2.0, lr * 0.1, lr * 0.2]:
                test_U = U - alpha * dU
                test_V = V - alpha * dV
                test_W = W - alpha * dW
                test_err = compute_error_fast(test_U, test_V, test_W)
                if test_err < best_step_err:
                    best_step_err = test_err
                    best_step_U, best_step_V, best_step_W = test_U, test_V, test_W
                    best_alpha = alpha
            
            U, V, W = best_step_U, best_step_V, best_step_W
            lr = best_alpha * 0.95
            
            # Check convergence
            err = best_step_err
            if err < tol:
                break
        
        return U, V, W
    
    def to_clean_list(mat):
        """Convert numpy array to list of lists with clean values."""
        result = []
        for row in mat:
            clean_row = []
            for x in row:
                # Try to find a simple fraction
                rounded = round(x)
                if abs(x - rounded) < 1e-9:
                    clean_row.append(int(rounded))
                else:
                    # Try halves
                    half = round(x * 2) / 2.0
                    if abs(x - half) < 1e-9:
                        clean_row.append(half)
                    else:
                        # Try quarters
                        quarter = round(x * 4) / 4.0
                        if abs(x - quarter) < 1e-9:
                            clean_row.append(quarter)
                        else:
                            # Try eighths
                            eighth = round(x * 8) / 8.0
                            if abs(x - eighth) < 1e-9:
                                clean_row.append(eighth)
                            else:
                                clean_row.append(float(x))
            result.append(clean_row)
        return result
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But the actual minimum rank might be lower due to the tensor structure
    # Let's try from 20 to 45
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    for r in range(20, 46):
        if time.time() > deadline - 15:
            break
        
        # Try multiple restarts
        found = False
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                # Use structured initialization
                U, V, W = try_structured_init(r)
            elif restart == 1:
                # Different structured init
                np.random.seed(77 + r)
                U = np.random.randn(n*m, r) * 0.7
                V = np.random.randn(m*p, r) * 0.7
                W = np.random.randn(p*n, r) * 0.7
                for c in range(r):
                    nu = norm(U[:, c])
                    nv = norm(V[:, c])
                    nw = norm(W[:, c])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        U[:, c] /= nu
                        V[:, c] /= nv
                        W[:, c] /= nw
            else:
                # Random initialization with varying scales
                scale = 0.2 + 0.1 * restart
                np.random.seed(42 + r * 100 + restart)
                U = np.random.randn(n*m, r) * scale
                V = np.random.randn(m*p, r) * scale
                W = np.random.randn(p*n, r) * scale
                
                # Normalize
                for c in range(r):
                    nu = norm(U[:, c])
                    nv = norm(V[:, c])
                    nw = norm(W[:, c])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        U[:, c] /= nu
                        V[:, c] /= nv
                        W[:, c] /= nw
            
            # Run gradient descent
            result = gradient_descent(r, max_iter=150, lr_init=0.02, lr_decay=0.998)
            
            if result is not None:
                U, V, W = result
                
                # Polish
                U, V, W = polish(U, V, W, max_iter=30, tol=1e-10)
                
                # Final check
                err = compute_error_fast(U, V, W)
                if err < 1e-7:
                    found = True
                    best_result = (U, V, W)
                    best_rank = r
                    break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
    
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
