"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin before 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 63) as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # i.e., T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid decomposition of the multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        # For each (i,j,k) triple, check the constraint
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], np.multiply(V[idx_v, :], W[idx_w, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iter=500, tol=1e-6):
        """Try to find a rank-r decomposition using random initialization and gradient descent."""
        if time.time() > deadline:
            return None
        
        # The tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
        # Number of constraints: n*m*p = 63
        # Unknowns: r*(n*m + m*p + p*n) = r*(9 + 21 + 21) = 51r
        
        # For rank r, we need 51r >= 63, so r >= 2 (but realistically much higher)
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Gradient descent on the loss: sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        lr = 0.01
        best_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        # val = sum_r U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                        val = np.dot(U[idx_u, :], np.multiply(V[idx_v, :], W[idx_w, :]))
                        err = val - 1.0
                        loss += err * err
                        
                        # dU[idx_u,r] += 2 * err * V[idx_v,r] * W[idx_w,r]
                        dU[idx_u, :] += 2 * err * np.multiply(V[idx_v, :], W[idx_w, :])
                        dV[idx_v, :] += 2 * err * np.multiply(U[idx_u, :], W[idx_w, :])
                        dW[idx_w, :] += 2 * err * np.multiply(U[idx_u, :], V[idx_v, :])
            
            if loss < best_loss:
                best_loss = loss
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if it > 0 and loss > best_loss * 1.1:
                lr *= 0.5
            
            # Check convergence
            if loss < 1e-10:
                break
        
        # Check if we found a valid solution
        if check_decomposition(U, V, W, tol=1e-4):
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        return None
    
    # Try ranks from low to high
    # Lower bound: rank >= max(n, m, p) = 7 for 3x3x7
    # But realistically, for matrix multiplication, rank is often close to n*m*p or somewhat less
    
    # Try a few promising ranks
    for r in [7, 8, 9, 10, 12, 15, 18, 21, 24, 27, 30, 33, 36, 40, 45, 50, 54, 57, 60, 63]:
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=200)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
