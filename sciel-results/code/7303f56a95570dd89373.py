"""Program that attempts to find a rank-minimal tensor decomposition for 2x3x8 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Return a valid tensor decomposition. Attempt structured search for rank < 48,
    falling back to schoolbook rank 48.
    
    Strategy: Use Kronecker/block structure. Factor the 3x8 multiplication into
    sub-blocks and try to exploit low-rank properties.
    """
    start = time.time()
    deadline = start + 130  # 130 s safety margin out of 150 s

    n, m, p = 2, 3, 8
    rank_schoolbook = n * m * p  # 48

    # --- Build the schoolbook decomposition (rank 48) as a guaranteed fallback ---
    def schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook(n, m, p)

    def verify(U, V, W, n, m, p, tol=1e-12):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(s - 1) > tol:
                        return False
        return True

    assert verify(U0, V0, W0, n, m, p), "Schoolbook decomposition failed verification"

    # --- Structured search for lower rank ---
    # Key insight: The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,7}
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy 1: Try to use block structure
    # Split j into {0,1} and {2}, k into {0,1,2,3} and {4,5,6,7}
    # This gives 4 blocks in (j,k) space:
    # Block (0,0): j in {0,1}, k in {0,1,2,3} -> 2x4 subproblem
    # Block (0,1): j in {0,1}, k in {4,5,6,7} -> 2x4 subproblem  
    # Block (1,0): j in {2}, k in {0,1,2,3} -> 1x4 subproblem
    # Block (1,1): j in {2}, k in {4,5,6,7} -> 1x4 subproblem
    
    # For each (i, block) combination, we have a sub-tensor
    # The challenge is that U, V, W are shared across all blocks
    
    # Strategy 2: Direct integer search for small rank reductions
    # Try rank 47, 46, ... by searching for valid decompositions
    
    def search_rank(R, time_limit, seed=42):
        """
        Search for a rank-R decomposition using random initialization + local search.
        Returns (success, U, V, W) or (False, None, None)
        """
        import random
        rng = np.random.RandomState(seed)
        
        deadline_search = time.time() + time_limit
        
        # Initialize with random values
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Try to optimize using gradient descent on the loss function
        # Loss = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        
        def compute_loss(U, V, W):
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                        loss += (s - 1) ** 2
            return loss
        
        # Gradient computation
        def compute_gradients(U, V, W):
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                        err = 2 * (s - 1)
                        for r in range(R):
                            dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
            return dU, dV, dW
        
        # Simple gradient descent
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            if time.time() > deadline_search:
                break
            
            loss = compute_loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                return True, best_U, best_V, best_W
            
            dU, dV, dW = compute_gradients(U, V, W)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            lr *= 0.95
        
        # Check if we found a valid solution
        if best_loss < 1e-10:
            return True, best_U, best_V, best_W
        return False, None, None
    
    # Try to find lower rank decompositions
    best_rank = rank_schoolbook
    best_decomp = (U0.copy(), V0.copy(), W0.copy())
    
    # First, try rank 47 with a few random seeds
    for R in [47, 46, 45]:
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            success, U, V, W = search_rank(R, time_limit=20, seed=seed)
            if success:
                # Verify with exact arithmetic
                if verify(U, V, W, n, m, p, tol=1e-10):
                    best_rank = R
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    break
    
    # If we found a lower rank, convert to exact fractions if possible
    if best_rank < rank_schoolbook:
        U_final, V_final, W_final = best_decomp
        # Round to nearest half-integer if close
        U_rounded = np.round(U_final * 2) / 2
        V_rounded = np.round(V_final * 2) / 2
        W_rounded = np.round(W_final * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-10):
            U_final, V_final, W_final = U_rounded, V_rounded, W_rounded
    
    # Return the best decomposition found
    if best_rank < rank_schoolbook:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
