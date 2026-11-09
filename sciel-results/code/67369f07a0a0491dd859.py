"""Matrix multiplication 2x3 * 3x6: rank-minimal tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                            expected = (1.0 if (a==i and c==b and d==k) else 0.0)
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_18(n, m, p, deadline):
    """
    Try to find a rank-18 decomposition using structured parameterization.
    
    For 2x3*3x6, we can use the fact that:
    - U has shape (6, 18)
    - V has shape (18, 18)  
    - W has shape (12, 18)
    
    We'll use a parameterized form and optimize.
    """
    rank = 18
    
    # Initialize with random values
    np.random.seed(42)
    
    # Create random initial matrices
    U_init = np.random.randn(n*m, rank)
    V_init = np.random.randn(m*p, rank)
    W_init = np.random.randn(p*n, rank)
    
    # Normalize to have reasonable scale
    U_init = U_init / np.sqrt(rank)
    V_init = V_init / np.sqrt(rank)
    W_init = W_init / np.sqrt(rank)
    
    # We need to solve: for all (a,b,c,d,i,k):
    # sum_t U[a*m+b,t] * V[c*p+d,t] * W[k*n+i,t] = delta(a,i)*delta(c,b)*delta(d,k)
    
    # This is equivalent to: the tensor T_{a,b,c,d,i,k} = sum_t U[a*m+b,t] * V[c*p+d,t] * W[k*n+i,t]
    # should equal the identity tensor.
    
    # Let's use gradient descent on the squared error
    # Error = sum_{a,b,c,d,i,k} (T_{a,b,c,d,i,k} - expected)^2
    
    # Flatten U, V, W into vectors
    U_flat = U_init.flatten()
    V_flat = V_init.flatten()
    W_flat = W_init.flatten()
    
    x = np.concatenate([U_flat, V_flat, W_flat])
    
    # Precompute indices
    n, m, p = 2, 3, 6
    
    # Number of constraints
    num_constraints = n * m * m * p * n * p  # a,b,c,d,i,k
    
    # For efficiency, we'll compute the error and gradient analytically
    # But for now, let's use a simpler approach with finite differences
    
    def compute_error(x_vec):
        U_v = x_vec[:n*m*rank].reshape(n*m, rank)
        V_v = x_vec[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_v = x_vec[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        error = 0.0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                s = 0.0
                                for t in range(rank):
                                    s += U_v[a*m+b, t] * V_v[c*p+d, t] * W_v[k*n+i, t]
                                expected = 1.0 if (a==i and c==b and d==k) else 0.0
                                error += (s - expected) ** 2
        return error
    
    def compute_gradient(x_vec):
        U_v = x_vec[:n*m*rank].reshape(n*m, rank)
        V_v = x_vec[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_v = x_vec[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        grad_U = np.zeros_like(U_v)
        grad_V = np.zeros_like(V_v)
        grad_W = np.zeros_like(W_v)
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                expected = 1.0 if (a==i and c==b and d==k) else 0.0
                                s = 0.0
                                for t in range(rank):
                                    s += U_v[a*m+b, t] * V_v[c*p+d, t] * W_v[k*n+i, t]
                                diff = s - expected
                                for t in range(rank):
                                    grad_U[a*m+b, t] += 2 * diff * V_v[c*p+d, t] * W_v[k*n+i, t]
                                    grad_V[c*p+d, t] += 2 * diff * U_v[a*m+b, t] * W_v[k*n+i, t]
                                    grad_W[k*n+i, t] += 2 * diff * U_v[a*m+b, t] * V_v[c*p+d, t]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    best_error = compute_error(x)
    best_x = x.copy()
    
    max_iterations = 5000
    for it in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        err = compute_error(x)
        if err < best_error:
            best_error = err
            best_x = x.copy()
        
        if best_error < 1e-12:
            break
        
        grad = compute_gradient(x)
        x = x - lr * grad
        
        # Adaptive learning rate
        new_err = compute_error(x)
        if new_err > err * 1.01:
            lr *= 0.5
            x = best_x  # rollback
        elif it % 100 == 0:
            lr *= 1.1
    
    # Extract best solution
    U_best = best_x[:n*m*rank].reshape(n*m, rank)
    V_best = best_x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_best = best_x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Convert to list of lists
    U_list = [[float(val) for val in row] for row in U_best]
    V_list = [[float(val) for val in row] for row in V_best]
    W_list = [[float(val) for val in row] for row in W_best]
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, rank
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with W[k*n+i, t] indexing.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook_decomposition(n, m, p)
    
    # Verify schoolbook is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen
        pass
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try rank 18
    if time.time() < deadline - 10:
        result = _try_rank_18(n, m, p, deadline)
        if result is not None:
            U18, V18, W18, rank18 = result
            if _verify_decomposition(U18, V18, W18, n, m, p):
                best_U, best_V, best_W, best_rank = U18, V18, W18, rank18
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
