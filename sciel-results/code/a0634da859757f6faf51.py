"""Program for matrix multiplication 3x3x7 tensor decomposition search."""
import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly reproduces matrix multiplication.
    
    The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all (i,j,k) in range(n) x range(m) x range(p).
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=0):
    """Generate a random initialization for the factor matrices."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _optimize_decomposition(n, m, p, rank, max_iters=200, tol=1e-10, seed=0, deadline=None):
    """Optimize factor matrices using gradient descent on the non-convex problem.
    
    We minimize sum_{i,j,k} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
    """
    rng = np.random.RandomState(seed)
    
    U, V, W = _random_decomposition(n, m, p, rank, seed)
    
    # Precompute indices
    indices = [(i, j, k, i * m + j, j * p + k, k * n + i) 
               for i in range(n) for j in range(m) for k in range(p)]
    
    best_loss = float('inf')
    lr = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k, u_idx, v_idx, w_idx) in indices:
            # Compute the dot product: sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
            val = 0.0
            for c in range(rank):
                val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            
            diff = 1.0 - val
            loss += diff * diff
            
            # Gradients
            for c in range(rank):
                # d/dU[u_idx,c] = -2 * diff * V[v_idx,c] * W[w_idx,c]
                grad_U[u_idx, c] += -2.0 * diff * V[v_idx, c] * W[w_idx, c]
                # d/dV[v_idx,c] = -2 * diff * U[u_idx,c] * W[w_idx,c]
                grad_V[v_idx, c] += -2.0 * diff * U[u_idx, c] * W[w_idx, c]
                # d/dW[w_idx,c] = -2 * diff * U[u_idx,c] * V[v_idx,c]
                grad_W[w_idx, c] += -2.0 * diff * U[u_idx, c] * V[v_idx, c]
        
        # Update with Adam-like optimization
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Reduce learning rate
        if it % 20 == 0 and it > 0:
            lr *= 0.95
        
        if loss < best_loss:
            best_loss = loss
        
        if loss < tol:
            break
    
    return U, V, W, best_loss

def _try_rank(n, m, p, rank, max_seeds=5, deadline=None):
    """Try to find a valid decomposition with the given rank."""
    for seed in range(max_seeds):
        if deadline is not None and time.time() > deadline:
            break
        U, V, W, loss = _optimize_decomposition(n, m, p, rank, max_iters=100, seed=seed, deadline=deadline)
        
        if loss < 1e-9:
            # Verify with exact arithmetic check
            U_list = [[float(x) for x in U[i]] for i in range(n * m)]
            V_list = [[float(x) for x in V[i]] for i in range(m * p)]
            W_list = [[float(x) for x in W[i]] for i in range(p * n)]
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = (schoolbook_rank, U_school, V_school, W_school)
    
    deadline = time.time() + 120
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= n*m*p / max(n,m,p) = 63/7 = 9
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank = max(n * m, m * p, p * n)
    
    # Try a range of ranks, starting from lower
    for rank in range(min_rank, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(n, m, p, rank, max_seeds=3, deadline=deadline)
        if result is not None:
            U, V, W = result
            best = (rank, U, V, W)
            break  # Found the minimum rank, stop
    
    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
