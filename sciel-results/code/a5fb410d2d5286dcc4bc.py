"""Optimized search for rank-minimal tensor decomposition of 2x4 * 4x4 matrix multiplication."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    n, m, p = 2, 4, 4
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult, T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
    # Actually the tensor for matrix mult C = A*B where A is n x m, B is m x p, C is n x p
    # T[i, j, k] = delta (the coefficient) - it's 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=2, m=4, p=4):
    """Faster verification using numpy."""
    n, m, p = 2, 4, 4
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(T, np.ones((n,m,p)), atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook rank 32
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank using randomized search with verification
    # The key insight: we need T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Strategy: try ranks from 16 upward (or from a reasonable lower bound)
    # Use random initialization and local search
    
    random.seed(42)
    
    # Lower bound: rank must be at least max(n*m, m*p, p*n) / something...
    # Actually, a known lower bound is max(n*m, m*p, p*n) but that's not tight.
    # For 2x4x4, let's try ranks from 12 to 31
    
    for target_rank in range(12, 32):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random values
            for c in range(target_rank):
                for idx in range(n*m):
                    U[idx, c] = random.uniform(-0.5, 0.5)
                for idx in range(m*p):
                    V[idx, c] = random.uniform(-0.5, 0.5)
                for idx in range(p*n):
                    W[idx, c] = random.uniform(-0.5, 0.5)
            
            # Local search: gradient-like updates
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] should be 1
                            # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                            # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                            # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                            error = 0.0
                            for c in range(target_rank):
                                val = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                error += val
                            err = error - 1.0
                            
                            for c in range(target_rank):
                                dU[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Found a solution!
                    # Round to nearest half-integer if possible
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded version
                    if _verify_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        best_rank = target_rank
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        return {"U": best_U, "V": best_V, "W": best_W}
                    else:
                        # Use unrounded but verify
                        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
                            best_rank = target_rank
                            best_U = U.tolist()
                            best_V = V.tolist()
                            best_W = W.tolist()
                            return {"U": best_U, "V": best_V, "W": best_W}
                    break
    
    # Return the best found (default is schoolbook)
    return {"U": best_U, "V": best_V, "W": best_W}
