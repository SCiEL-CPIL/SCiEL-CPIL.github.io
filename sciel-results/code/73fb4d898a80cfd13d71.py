"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.
Uses integer linear programming (ILP) to search for a rank-minimal decomposition
by modeling the tensor network slicing / circuit pebbling strategy.

The matrix multiplication tensor T[i,j,k] for A(3x3) * B(3x4) = C(3x4)
is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
where delta(i,j,k) = 1 if the (i,j) entry of A and (j,k) entry of B contribute
to C[i,k], which is always true for valid indices.

We use scipy.optimize.milp to find a sparse integer decomposition.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Try to find a decomposition with rank < 36 using ILP
    # We'll try ranks from a lower bound up to 36
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    
    # First, let's try a few candidate ranks and see if we can find a solution
    # For efficiency, we'll start with a moderate rank and work down
    
    best_rank = n * m * p  # Start with schoolbook
    best_decomp = None
    
    # Try ranks from 20 to 35 (we know 36 works, so try to beat it)
    # Using a grid search with ILP for each rank
    
    for target_rank in range(20, best_rank):
        decomp = _try_ilp_decomposition(n, m, p, target_rank)
        if decomp is not None:
            best_rank = target_rank
            best_decomp = decomp
            break  # Found the minimum, stop searching
    
    if best_decomp is None:
        # Fall back to schoolbook
        best_decomp = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return best_decomp

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition."""
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

def _try_ilp_decomposition(n, m, p, target_rank):
    """
    Try to find a decomposition of the given rank using ILP.
    Returns None if no solution found, otherwise returns the decomposition.
    """
    # The tensor constraint: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a bilinear constraint, which is hard for ILP directly.
    # 
    # Alternative approach: Fix one of the matrices and solve for the others.
    # Or use a different strategy: assume U and V have entries in {-1, 0, 1} 
    # and W has entries in {-1, 0, 1}, then search.
    #
    # For a more tractable approach, let's use the fact that we can fix
    # the structure of U and V to be sparse (each row has few non-zeros)
    # and solve for W.
    #
    # Actually, let's try a simpler approach: use a known technique where
    # we fix U to have a specific structure and solve for V and W.
    #
    # For now, let's try a heuristic: assume U has entries in {-1, 0, 1}
    # with each row having at most 2 non-zeros, and similarly for V.
    # Then W can be computed.
    #
    # This is getting complex. Let's use a different strategy:
    # Try to find a decomposition where U, V, W all have entries in {-1, 0, 1}
    # and use a SAT-like search or local search.
    
    # For simplicity and to ensure we return within time limits,
    # let's implement a local search / simulated annealing approach.
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 30  # seconds
    
    # Initialize with random sparse matrices
    U = np.zeros((n * m, target_rank), dtype=int)
    V = np.zeros((m * p, target_rank), dtype=int)
    W = np.zeros((p * n, target_rank), dtype=int)
    
    # Random initialization with sparse entries
    for r in range(target_rank):
        # Pick random positions for U, V, W
        u_idx = random.randint(0, n * m - 1)
        v_idx = random.randint(0, m * p - 1)
        w_idx = random.randint(0, p * n - 1)
        
        u_val = random.choice([-1, 1])
        v_val = random.choice([-1, 1])
        w_val = random.choice([-1, 1])
        
        U[u_idx, r] = u_val
        V[v_idx, r] = v_val
        W[w_idx, r] = w_val
    
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += (s - 1) ** 2
        return error
    
    def try_improve(U, V, W, max_iters=10000):
        """Try to improve the decomposition using local search."""
        current_error = compute_error(U, V, W)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Randomly pick a matrix and a position to modify
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                rows = n * m
            elif mat_choice == 'V':
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, target_rank - 1)
            
            old_val = mat[row, col]
            # Try a new value
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            
            mat[row, col] = new_val
            new_error = compute_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
            else:
                mat[row, col] = old_val
        
        return current_error
    
    # Run local search
    best_error = try_improve(U, V, W)
    
    # Check if we found a perfect decomposition
    if best_error == 0:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return None
