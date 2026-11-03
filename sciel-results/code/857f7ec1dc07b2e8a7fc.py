"""Matrix multiplication 2,3,7 - search for low-rank exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-42 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
            return False
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _to_fraction_list(mat):
    """Convert a numpy array to list of lists of Fractions."""
    return [[Fraction(int(round(x * 1000000)) / 1000000) for x in row] for row in mat]

def _try_rank(rank, max_iter=50, seed=0):
    """
    Try to find a rank-`rank` decomposition using random linear combinations
    and solving the resulting system.
    Returns (U, V, W) as lists of lists of Fractions, or None if not found.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_iter):
        # Random scaling of the tensor slices
        # We want to find U, V, W such that T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
        
        # Strategy: pick random linear combinations to create a system
        # For each rank-1 term, U[:, r] * V[:, r] * W[:, r] should contribute to T
        
        # Let's try a different approach: 
        # Assume we have rank r decomposition. Pick random vectors a (length n*m), 
        # b (length m*p), and try to find W such that...
        # This is getting complex. Let me try a simpler heuristic.
        
        # Actually, for small dimensions, let's try to find a decomposition
        # by assuming a specific structure and solving.
        
        # For 2x3x7, the schoolbook rank is 42. The lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
        # Known results: for 2x3x7, the rank is likely 21 or higher.
        
        # Let's try to find a rank-21 decomposition by exploiting the structure.
        # The multiplication of a 2x3 matrix by a 3x7 matrix can be seen as:
        # For each of the 7 columns of the second matrix, we do a 2x3 * 3x1 multiplication.
        # A 2x3 * 3x1 multiplication has rank at least max(2*3, 3*1, 1*2) = 6.
        # But we can share computations across columns.
        
        # Actually, let's just try random restarts with a simple optimization.
        # Initialize U, V, W randomly and try to optimize.
        
        U_np = rng.randn(n * m, rank) * 0.1
        V_np = rng.randn(m * p, rank) * 0.1
        W_np = rng.randn(p * n, rank) * 0.1
        
        # Try a few Newton steps or gradient descent
        for step in range(20):
            # Compute current tensor
            T_current = np.zeros((n, m, p))
            for r in range(rank):
                U_r = U_np[:, r].reshape(n, m)
                V_r = V_np[:, r].reshape(m, p)
                W_r = W_np[:, r].reshape(p, n)
                # T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_current[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
            
            # Target tensor
            T_target = np.ones((n, m, p))
            error = np.linalg.norm(T_target - T_current)
            if error < 1e-6:
                break
            
            # Simple gradient step
            lr = 0.01 / (1 + step)
            for r in range(rank):
                U_r = U_np[:, r].reshape(n, m)
                V_r = V_np[:, r].reshape(m, p)
                W_r = W_np[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = T_target[i, j, k] - T_current[i, j, k]
                            U_np[i * m + j, r] += lr * diff * V_r[j, k] * W_r[k, i]
                            V_np[j * p + k, r] += lr * diff * U_r[i, j] * W_r[k, i]
                            W_np[k * n + i, r] += lr * diff * U_r[i, j] * V_r[j, k]
        
        # Check if we converged
        T_current = np.zeros((n, m, p))
        for r in range(rank):
            U_r = U_np[:, r].reshape(n, m)
            V_r = V_np[:, r].reshape(m, p)
            W_r = W_np[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_current[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
        
        if np.linalg.norm(T_target - T_current) < 1e-4:
            # Try to rationalize
            U_frac = _to_fraction_list(U_np)
            V_frac = _to_fraction_list(V_np)
            W_frac = _to_fraction_list(W_np)
            if _verify(U_frac, V_frac, W_frac):
                return U_frac, V_frac, W_frac
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x7 matrix. Falls back to schoolbook if no better solution is found.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Try ranks from the theoretical lower bound up to schoolbook
    lower_bound = max(n * m, m * p, p * n)  # 21
    
    # Try a few ranks below schoolbook
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 30:  # Leave 30 seconds for finalization
            break
        result = _try_rank(rank, max_iter=10, seed=rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
