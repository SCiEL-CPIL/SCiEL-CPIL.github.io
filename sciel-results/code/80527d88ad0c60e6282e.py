"""Rewritten program for matrix multiplication 3x4x5."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Trivial decomposition: rank = n*m*p = 60
    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, best_rank = trivial_decomposition()
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to find better decompositions using structured search
    # The tensor T_{i,j,k} = delta_{i,j,k} for matrix multiplication
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 if valid, 0 otherwise
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Also rank >= n*m*p / min(n,m,p) = 60/3 = 20
    # Try ranks from 20 upward
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        # Compute the tensor from the decomposition
        # T[i*m+j, j*p+k, k*n+i] should be 1 for valid (i,j,k), 0 otherwise
        # Using einsum: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3-index tensor
        # Let's compute it efficiently
        rank = U.shape[1]
        # Reshape to get the tensor
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # We need to compute T[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # where a = i*m+j, b = j*p+k, c = k*n+i
        # And T should be 1 iff the indices are consistent (same i,j,k)
        
        # Build the target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    target[a, b, c] = 1.0
        
        # Compute the actual tensor from decomposition
        # Use batch matrix multiplication
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U[a,:]*V[b,:]) * W[c,r]
        # Let M = U @ V.T, shape (n*m, m*p)
        # Then T[a,b,c] = sum_r M[a,b] * W[c,r] ... no that's not right
        
        # Actually: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Let's compute it as: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # But that's expensive. Instead:
        # T = sum_r U[:,r] @ V[:,r].T @ W[:,r].T ... no
        
        # Let's use: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U @ V.T) doesn't directly help
        # We can compute: for each c, T[:,:,c] = sum_r (U[:,r]*V[:,r].T) * W[c,r]
        # Let's do it in a loop over r for clarity, or use einsum
        
        # Using einsum: T = np.einsum('ar,br,cr->abc', U, V, W)
        try:
            actual = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return False
        
        # Check if actual matches target
        if not np.allclose(actual, target, atol=tol):
            return False
        return True
    
    # Try structured decompositions
    # Strategy: use the fact that matrix multiplication can be decomposed
    # using sub-matrix multiplications
    
    # Attempt 1: Direct sum approach
    # Split the 3x4x5 multiplication into smaller blocks
    
    # For 3x4 * 4x5:
    # We can write A = [A1; A2; A3] where A1 is 1x4, A2 is 1x4, A3 is 1x4
    # B = [B1, B2, B3, B4] where each Bj is 4x1
    # C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try rank-55: maybe we can save 5 terms
    # One approach: use the structure that some terms can be shared
    
    # Let's try a more systematic search
    # For small ranks, use random initialization and gradient descent
    
    def try_rank(rank, max_iters=1000, lr=0.01):
        """Try to find a decomposition of given rank using gradient descent."""
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            actual = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((actual - target)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_loss < 1e-15:
                    break
            
            # Compute gradients
            # dL/dU[a,r] = 2 * sum_{b,c} (actual[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} (actual[a,b,c] - target[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} (actual[a,b,c] - target[a,b,c]) * U[a,r] * V[b,r]
            
            diff = actual - target
            
            # dL/dU = 2 * diff @ (V[:,r] * W[:,r]) ... need to be careful
            # dL/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # = 2 * (diff @ (V[:,r] * W[:,r]))[a] for each r
            # Let's compute it as: for each r, dU[:,r] = 2 * diff @ (V[:,r] * W[:,r])
            
            # Efficient computation:
            # dU = 2 * einsum('abc,br,cr->ar', diff, V, W)
            # dV = 2 * einsum('abc,ar,cr->br', diff, U, W)
            # dW = 2 * einsum('abc,ar,br->cr', diff, U, V)
            
            dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-10:
            return best_U, best_V, best_W, best_loss
        return None
    
    # Try various ranks
    for rank in range(20, 60):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        n_restarts = 3 if rank < 40 else 1
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, max_iters=500, lr=0.05)
            if result is not None:
                U, V, W, loss = result
                if loss < 1e-10:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        if best_rank < 60:
            break
    
    return best_result
# EVOLVE-BLOCK-END
