"""Rewritten program for matrix multiplication 3x4x5."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import random
from fractions import Fraction

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
    
    # Build target tensor once
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W form a valid decomposition."""
        try:
            actual = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return False
        if not np.allclose(actual, target, atol=tol):
            return False
        return True
    
    def try_rank_gradient(rank, max_iters=2000, lr=0.02, n_restarts=5):
        """Try to find a decomposition of given rank using gradient descent with multiple restarts."""
        if time.time() > deadline:
            return None
        
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with different scales
            scale = 0.1 * (1 + restart * 0.5)
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            current_loss = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                actual = np.einsum('ar,br,cr->abc', U, V, W)
                diff = actual - target
                loss = np.sum(diff**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_loss < 1e-14:
                        return best_U, best_V, best_W, best_loss
                
                # Adaptive learning rate
                effective_lr = lr * (1.0 / (1.0 + it * 0.01))
                
                # Compute gradients
                dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Gradient step with momentum-like behavior
                U -= effective_lr * dU
                V -= effective_lr * dV
                W -= effective_lr * dW
        
        if best_loss < 1e-10 and best_U is not None:
            return best_U, best_V, best_W, best_loss
        return None
    
    def try_rank_structured(rank):
        """Try structured decompositions based on block structure."""
        if time.time() > deadline:
            return None
        
        # For 3x4*4x5, we can try various block decompositions
        # Strategy: split into smaller matrix multiplications
        
        # Approach 1: Direct sum of 1x1 multiplications (rank 60)
        # Approach 2: Use 2x2 blocks where possible
        
        # Try to find a structured decomposition
        # The key insight: C[i,k] = sum_j A[i,j]*B[j,k]
        # We can group terms to reduce rank
        
        # For now, return None and rely on gradient descent
        return None
    
    # Try various ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Also: rank >= n*m*p / max(n,m,p) = 60/5 = 12, but structure gives 20
    
    # Try ranks 55, 56, 57, 58, 59 (near trivial) first with more effort
    # Then try lower ranks with less effort
    
    ranks_to_try = [55, 56, 57, 58, 59, 50, 52, 54, 45, 48, 40, 35, 30, 25, 22, 20]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try structured first
        structured = try_rank_structured(rank)
        if structured is not None:
            U, V, W, loss = structured
            if verify(U, V, W):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        # Try gradient descent with multiple restarts
        n_restarts = 10 if rank >= 55 else 5 if rank >= 45 else 3
        max_iters = 1500 if rank >= 55 else 1000 if rank >= 45 else 500
        
        result = try_rank_gradient(rank, max_iters=max_iters, lr=0.02, n_restarts=n_restarts)
        if result is not None:
            U, V, W, loss = result
            if verify(U, V, W):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    # Final check: if we found a better rank, verify it one more time
    if best_rank < 60:
        U = np.array(best_result["U"])
        V = np.array(best_result["V"])
        W = np.array(best_result["W"])
        if not verify(U, V, W):
            # Fallback to trivial
            best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
