"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find minimal rank decomposition
    # Start with a reasonable lower bound and increase
    # For n*m*p matrix multiplication, the rank is at least max(n*m, m*p, n*p)
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to n*m*p
    for rank in range(lower_bound, n*m*p + 1):
        if _try_decomposition(n, m, p, rank):
            return _get_best_decomposition(n, m, p, rank)
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_decomposition(n, m, p, rank):
    """Try to find a rank-decomposition for n*m x m*p matrix multiplication"""
    # The tensor T has shape (n*m, m*p, n*p) in the appropriate indexing
    # We want U (n*m x rank), V (m*p x rank), W (n*p x rank) such that
    # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # With the trace convention: W[k*n+i, r]
    
    # Use random initialization and optimization
    best_solution = None
    best_error = float('inf')
    
    for trial in range(20):
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = np.random.randint(-2, 3, size=(n*p, rank))
        
        # Check if this gives an exact decomposition
        error = _compute_error(U, V, W, n, m, p)
        if error < 1e-6:
            best_solution = (U, V, W)
            best_error = error
            break
        elif error < best_error:
            best_error = error
            best_solution = (U, V, W)
    
    # Also try with half-integers (multiply by 2)
    for trial in range(20):
        U = np.random.randint(-4, 5, size=(n*m, rank))
        V = np.random.randint(-4, 5, size=(m*p, rank))
        W = np.random.randint(-4, 5, size=(n*p, rank))
        
        error = _compute_error(U, V, W, n, m, p)
        if error < 1e-6:
            # Convert to half-integers
            U_half = U / 2.0
            V_half = V / 2.0
            W_half = W / 2.0
            best_solution = (U_half, V_half, W_half)
            best_error = error
            break
        elif error < best_error:
            best_error = error
            best_solution = (U, V, W)
    
    if best_solution is not None and best_error < 1e-6:
        # Store the best solution
        global _best_decomposition
        _best_decomposition = best_solution
        return True
    return False

def _compute_error(U, V, W, n, m, p):
    """Compute the error of the decomposition"""
    # U: (n*m, rank), V: (m*p, rank), W: (n*p, rank)
    # The tensor should be: sum_r U[:,r] outer V[:,r] outer W[:,r]
    # For matrix multiplication, T[i,j,k] = 1 if i is row of A, j is col of A/row of B, k is col of B
    # In our indexing: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    rank = U.shape[1]
    # Compute the tensor
    T_computed = np.zeros((n*m, m*p, n*p))
    for r in range(rank):
        T_computed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # The target tensor
    T_target = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1
    
    return np.linalg.norm(T_computed - T_target)

def _get_best_decomposition(n, m, p, rank):
    """Get the best decomposition found"""
    global _best_decomposition
    if _best_decomposition is not None:
        U, V, W = _best_decomposition
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    return _schoolbook(n, m, p)

# Global variable to store best decomposition
_best_decomposition = None
# EVOLVE-BLOCK-END
