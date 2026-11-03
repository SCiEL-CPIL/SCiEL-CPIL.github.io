import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s margin
    
    n, m, p = 3, 3, 4
    N = n * m
    M = m * p
    P = p * n
    
    # Precompute indices for the tensor condition
    # We need to check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i, j, k
    
    # Schoolbook solution
    def schoolbook():
        R = n * m * p
        U = np.zeros((N, R), dtype=int)
        V = np.zeros((M, R), dtype=int)
        W = np.zeros((P, R), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {'U': U, 'V': V, 'W': W}
    
    # Check if a candidate solution is valid
    def is_valid(U, V, W):
        # U: (N, R), V: (M, R), W: (P, R)
        R = U.shape[1]
        # Construct the tensor contraction
        # For each r, we have a term U_r * V_r * W_r
        # We need the sum over r to be 1 at (i,j,k)
        
        # Efficient check:
        # Create a 3D array T of shape (n, m, p) initialized to 0
        T = np.zeros((n, m, p), dtype=float)
        
        for r in range(R):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            
            # Reshape to matrices
            U_mat = u.reshape(n, m)
            V_mat = v.reshape(m, p)
            W_mat = w.reshape(p, n).T # W index is k*n+i, so W_mat[k, i] is w[k*n+i]
            # Wait, W shape is (P, R). P = p*n.
            # Index k*n+i.
            # If we reshape w to (p, n), then element [k, i] is w[k*n+i]. Correct.
            
            # The term contributed by rank r is:
            # (A * U_mat^T?) No.
            # The bilinear form is sum_j A_ij U_ij^r * sum_k B_jk V_jk^r * W_ik^r?
            # No, the tensor entry is U_ij^r * V_jk^r * W_ik^r.
            # We are summing these over r.
            # So T[i, j, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
            
            # We can compute this using einsum or loops
            # Since n,m,p are small, a loop is fine, but vectorized is better.
            # T += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
            
            T += np.einsum('ij,jk,ki->ijk', U_mat, V_mat, W_mat)
        
        # Check if T is all ones
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Search for lower rank
    # Try R from 22 to 35
    for R in range(22, 36):
        if time.time() > DEADLINE:
            break
        
        # Random search with local improvements
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            
            # Initialize with random integers {-1, 0, 1}
            # Sparsity can help
            U = np.random.choice([-1, 0, 1], size=(N, R))
            V = np.random.choice([-1, 0, 1], size=(M, R))
            W = np.random.choice([-1, 0, 1], size=(P, R))
            
            # Local search: try to reduce error by changing one element
            # Error = sum (T[i,j,k] - 1)^2
            # This is expensive to compute fully each step.
            # Instead, just check validity occasionally.
            
            # For this simple rewrite, we'll just do random restarts.
            # If R=22 is hard to find randomly, we might not find it.
            
            if is_valid(U, V, W):
                return {'U': U, 'V': V, 'W': W}
    
    return schoolbook()
