"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    
    # Schoolbook decomposition as baseline
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # The tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} for matrix mult
        # Actually for matrix mult: T[i,j,k,l] where C[i,l] = sum_j A[i,j]*B[j,l]
        # Tensor form: T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = delta_{j,j} (always 1 in the middle index sense)
        # More precisely: the tensor is T[i,j,k] = 1 for all i,j,k in the appropriate ranges
        # But that's not right. Let me think again.
        #
        # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The bilinear form: T(A,B)[i,k] = sum_j A[i,j] B[j,k]
        # As a tensor: T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
        # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid i,j,k
        # Wait, the output index is (i,k) and the convention is W[k*n+i, r]
        # So W index is k*n + i, meaning W is indexed by (k,i)
        # 
        # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) and is indexed by [k*n + i, r]
        # 
        # The decomposition should satisfy:
        # For all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # This is because the tensor T[i,j,k] = 1 for matrix multiplication.
        
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=5000, lr=0.01, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Gradient descent
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] should equal 1
                        # val = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        uv = np.multiply(U[i*m+j], V[j*p+k])
                        val = np.dot(uv, W[k*n+i])
                        loss += (val - 1.0) ** 2
                        
                        # d/dU[i*m+j,r] = 2*(val-1) * V[j*p+k,r] * W[k*n+i,r]
                        dU = 2.0 * (val - 1.0) * np.multiply(V[j*p+k], W[k*n+i])
                        grad_U[i*m+j] += dU
                        
                        # d/dV[j*p+k,r] = 2*(val-1) * U[i*m+j,r] * W[k*n+i,r]
                        dV = 2.0 * (val - 1.0) * np.multiply(U[i*m+j], W[k*n+i])
                        grad_V[j*p+k] += dV
                        
                        # d/dW[k*n+i,r] = 2*(val-1) * U[i*m+j,r] * V[j*p+k,r]
                        dW = 2.0 * (val - 1.0) * np.multiply(U[i*m+j], V[j*p+k])
                        grad_W[k*n+i] += dW
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-12:
                break
        
        # Check if we found a valid decomposition
        if verify_decomposition(U, V, W, tol=1e-8):
            return (U, V, W)
        
        return None
    
    # Try ranks from low to high
    for r in range(1, rank_schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        if r >= best_rank:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(r, max_iters=3000, lr=0.005, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
