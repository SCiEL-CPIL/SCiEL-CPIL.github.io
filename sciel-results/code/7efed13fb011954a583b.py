"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_decomposition(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using alternating minimization"""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Alternating least squares approach
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute predictions
            # pred[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Residual = 1 - pred
            
            # Update U: for each (i,j), solve min_U sum_k (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            # This is: min_U sum_r U[i*m+j,r] * A_{ij}[r] + B_{ij}
            # where A_{ij}[r] = sum_k V[j*p+k,r]*W[k*n+i,r] * pred[i,j,k]
            # and B_{ij} = sum_k V[j*p+k,r]*W[k*n+i,r]
            
            # For fixed i,j: minimize sum_k (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
            # Let S_{ij}[r] = sum_k V[j*p+k,r]*W[k*n+i,r]
            # Let P_{ij}[r] = sum_k pred[i,j,k]*V[j*p+k,r]*W[k*n+i,r]
            # Gradient wrt U[i*m+j,r] = -2*(1 - pred[i,j,k])*V[j*p+k,r]*W[k*n+i,r] summed over k
            
            # Actually, let's use a simpler gradient descent approach
            # Compute pred
            pred = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pred[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 1.0 - pred[i, j, k]
                        grad_U[i*m+j, :] -= 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] -= 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] -= 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                loss = np.sum((pred - 1.0)**2)
                if loss < 1e-12:
                    break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
        
        return None
    
    def try_structured_decomposition():
        """Try a structured decomposition using block matrix multiplication"""
        if time.time() > deadline:
            return None
        
        # Strategy: decompose 3x5x6 into blocks
        # A is 3x5, B is 5x6, C is 3x6
        # Split A into rows, B into columns, etc.
        
        # Try: use 2x2x2 Strassen-like decomposition for sub-blocks
        # This is complex, so let's try a simpler approach
        
        # Direct structured approach:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We can try to use the fact that n=3 is small
        
        # For each pair (i,k), we need sum_j A[i,j]*B[j,k]
        # This is an inner product of length m=5
        
        # Rank-1 term: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For a fixed r, this contributes U[i,j,r]*V[j,k,r]*W[k,i,r] to C[i,k]
        
        # Let's try a decomposition where we use m=5 terms for the j-sum
        # Plus some correction terms
        
        # Actually, let's try a more systematic approach:
        # Use the fact that we can write the multiplication as a sum of rank-1 tensors
        # where each rank-1 tensor corresponds to a "path" through the indices
        
        # For now, return schoolbook
        return schoolbook()
    
    # First, try to find a good decomposition using various strategies
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try structured decomposition
    structured = try_structured_decomposition()
    if structured is not None:
        r_struct = len(structured['U'][0])
        if r_struct < best_rank and verify(np.array(structured['U']), np.array(structured['V']), np.array(structured['W']), n, m, p):
            best_result = structured
            best_rank = r_struct
    
    # Try gradient-based optimization for various ranks
    # Start from the lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = try_decomposition(r, max_iters=2000, seed=seed)
            if result is not None:
                best_result = result
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END

