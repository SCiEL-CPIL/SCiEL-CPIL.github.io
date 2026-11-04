"""Optimized search for low-rank tensor decomposition of 4x4 * 4x9 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]
DEADLINE = 120.0  # seconds, with safety margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N = n * m  # 16
    M = m * p  # 36
    P = p * n  # 36
    
    # The tensor T has shape (N, M, P) where T[i, j, k] = 1 if the output index corresponds to input indices
    # Actually, the matrix multiplication tensor T_{ijk} = A_{ij} B_{jk} C_{ki}... 
    # Wait, the standard tensor for matrix multiplication C = AB is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has entries T[i, j, k] = delta_{ik} (in the sense that the output C_{ik} is produced by summing over j)
    # More precisely, the multiplication tensor M_{n,m,p} is a 3-tensor of size (n*p, m*p, n*m) or similar.
    # Let's use the convention from the task:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)  
    # W has shape (p*n, rank)
    # The decomposition is: for each output element C_{k*n+i} (using the W[k*n+i, r] convention),
    # C_{k*n+i} = sum_r U[i*m+j... wait, let me re-read.
    
    # The schoolbook decomposition has:
    # U[i*m + j, column] = 1
    # V[j*p + k, column] = 1
    # W[k*n + i, column] = 1
    # for each (i, j, k) triple, with column = i*m*p + j*p + k
    
    # So the tensor being decomposed has entries:
    # T[a, b, c] where a = i*m+j (input A index), b = j*p+k (input B index), c = k*n+i (output C index)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i, j, k
    # Otherwise T[a, b, c] = 0
    
    # This is the standard matrix multiplication tensor.
    # We want to find rank-minimal U, V, W such that:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Let's construct the tensor T
    T = np.zeros((N, M, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook rank is N*M/P... no, it's n*m*p = 144
    schoolbook_rank = n * m * p  # 144
    
    # We'll use a random initialization and gradient descent approach
    # to try to find a lower rank decomposition.
    
    def time_left():
        return DEADLINE - (time.time() - start_time) - 5.0  # 5 sec safety margin
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a tensor contraction
        # U: (N, rank), V: (M, rank), W: (P, rank)
        # We can compute this as: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        # More efficiently: 
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol, rtol=tol)
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random init + gradient descent."""
        if time_left() < 10.0:
            return None
        
        np.random.seed(42 + rank)
        
        # Initialize with small random values
        U = np.random.randn(N, rank) * 0.1
        V = np.random.randn(M, rank) * 0.1
        W = np.random.randn(P, rank) * 0.1
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        max_iter = 2000
        for it in range(max_iter):
            if time_left() < 10.0:
                break
            
            # Compute gradient of loss = 0.5 * ||T_recon - T||^2
            # T_recon = einsum('ar,br,cr->abc', U, V, W)
            # dL/dU = einsum('abcr,br,cr->ar', T_recon - T, V, W)
            # Actually: dL/dU[a,r] = sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_recon - T
            
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            loss = 0.5 * np.sum(diff**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Adaptive learning rate
            if it % 100 == 0:
                if loss < 1e-4:
                    lr = 0.001
                elif loss < 1e-2:
                    lr = 0.005
                else:
                    lr = 0.01
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if verify(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from low to high
    # Known lower bound for 4x4*4x9: at least max(n*p, m*p, n*m) = max(36, 36, 16) = 36
    # But realistically, it's likely around 100-144
    
    best_result = None
    for rank in range(80, schoolbook_rank + 1):
        if time_left() < 15.0:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, rank)
            break
    
    # If we found a result, convert to lists
    if best_result is not None:
        U, V, W, rank = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = schoolbook_rank
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
