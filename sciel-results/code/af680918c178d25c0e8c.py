"""Tensor decomposition for 3x4 by 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Build the tensor T[i, j, k] = 1 for all i, j, k
    # We want to find rank r such that T = sum_c U[:,c] V[:,c] W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The tensor element T[(i,j),(j,k),(k,i)] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook gives rank = n*m*p = 72
    # We'll try to find a lower rank decomposition
    
    # Strategy: Use the fact that for matrix multiplication, there are known
    # lower bounds. For 3x4 by 4x6, the rank is at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # and at most n*m*p = 72.
    
    # Let's try to use a random linear combination approach to find linear
    # dependencies among the rank-1 terms.
    
    # First, let's try small ranks with random search
    # For each rank r, we try to find U, V, W such that the tensor equation holds.
    
    # The tensor equation can be written as:
    # For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
    
    # We can use a flattening approach: fix one index and get a matrix equation.
    # For fixed i: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j, k
    # This gives: (V * W_i) * U_i = 1 matrix? No, it's more complex.
    
    # Alternative: use the "border rank" approach or direct optimization.
    
    # Let's try a simple approach: start with rank = max(n*m, m*p, p*n) + 1
    # and try to reduce.
    
    # Actually, let's just try to verify if rank = n*m*p - 1 works, etc.
    # But that's hard to check directly.
    
    # Better approach: use random projections to detect linear dependencies.
    
    # The tensor has 72 rank-1 terms in the schoolbook decomposition.
    # If there are linear dependencies, we can reduce the rank.
    
    # For matrix multiplication, the tensor T_{n,m,p} has a known structure.
    # The rank is at least max(n*m, m*p, p*n) and at most n*m*p.
    
    # For 3x4 by 4x6: max(12, 24, 18) = 24, n*m*p = 72
    
    # Let's try to find the actual rank using a simple randomized approach.
    # We'll use the fact that if we can find r vectors u_c, v_c, w_c such that
    # sum_c u_c(i,j) v_c(j,k) w_c(k,i) = 1 for all i,j,k, then rank <= r.
    
    # Start with the schoolbook decomposition and try to eliminate terms.
    
    # Schoolbook: for each (i,j,k), we have a term with U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # where c = i*m*p + j*p + k
    
    # We want to find linear combinations of these terms that sum to zero.
    
    # Let's try a different approach: use random matrices to test if a given rank works.
    
    def test_rank(r, max_attempts=100):
        """Try to find a decomposition of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Use gradient descent to minimize the error
            # Error = sum over all (i,j,k) of (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    return None
                
                # Compute gradient
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # f = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # df/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                            # df/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                            # df/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                            f = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = f - 1.0
                            U_grad[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                            V_grad[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                            W_grad[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence
                if step % 200 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                f = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                max_err = max(max_err, abs(f - 1.0))
                    if max_err < 1e-6:
                        return (U, V, W)
        return None
    
    # Try ranks from lower to upper
    min_rank = max(n*m, m*p, p*n)
    max_rank = n * m * p
    
    best_result = None
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 10:
            break
        result = test_rank(r, max_attempts=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    else:
        U, V, W = best_result
        rank = U.shape[1]
    
    # Convert to lists of integers/half-integers if possible
    def to_list(mat):
        lst = []
        for row in mat:
            lst.append(list(row))
        return lst
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
