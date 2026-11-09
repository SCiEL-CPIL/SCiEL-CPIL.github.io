"""
Tensor decomposition for 3x4x4 matrix multiplication.
Attempts to find rank < 48 using optimization, falls back to schoolbook rank 48.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

def _build_tensor_constraints():
    """Build the list of (a,b,c, target) constraints for the tensor."""
    n, m, p = DIMENSIONS
    constraints = []
    # Active constraints: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                constraints.append((a, b, c, 1))
    # Inactive constraints: all other (a,b,c) must be 0
    active_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_set.add((a, b, c))
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in active_set:
                    constraints.append((a, b, c, 0))
    return constraints

def _residual(U, V, W, constraints):
    """Compute the residual (max absolute error) for given factor matrices."""
    max_err = 0.0
    for (a, b, c, target) in constraints:
        val = np.dot(U[a], V[b] * W[c])
        err = abs(val - target)
        if err > max_err:
            max_err = err
    return max_err

def _try_rank(rank, n, m, p, constraints, max_iter=2000, time_limit=100):
    """Try to find a decomposition of given rank using gradient descent."""
    start_time = time.time()
    
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient numerically or analytically
        # For efficiency, compute residual and use simple gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, c, target) in constraints:
            val = np.dot(U[a], V[b] * W[c])
            err = val - target
            # d/dU[a] = V[b] * W[c] * err
            grad_U[a] += V[b] * W[c] * err
            # d/dV[b] = U[a] * W[c] * err
            grad_V[b] += U[a] * W[c] * err
            # d/dW[c] = U[a] * V[b] * err
            grad_W[c] += U[a] * V[b] * err
        
        # Gradient descent step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        err = _residual(U, V, W, constraints)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
        
        # Adaptive learning rate
        if err > best_err:
            lr *= 0.5
        elif err < best_err * 0.5:
            lr = min(lr * 1.1, 0.1)
    
    if best_err < 1e-6:
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        err_rounded = _residual(U_rounded, V_rounded, W_rounded, constraints)
        if err_rounded < 1e-8:
            return U_rounded, V_rounded, W_rounded, True
        else:
            return best_U, best_V, best_W, False
    
    return None, None, None, False

def _schoolbook():
    """Return the standard schoolbook decomposition with rank 48."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, constraints):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for (a, b, c, target) in constraints:
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(len(U[0])))
        err = abs(val - target)
        if err > max_err:
            max_err = err
    return max_err < 1e-10

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a decomposition for 3x4x4 matrix multiplication.
    Attempts to find rank < 48, falls back to schoolbook rank 48.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    n, m, p = DIMENSIONS
    
    constraints = _build_tensor_constraints()
    
    # Try to find rank < 48
    for rank in range(47, 40, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, constraints, max_iter=1000, time_limit=deadline - time.time() - 5)
        if result[3]:
            U, V, W, _ = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, constraints):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
