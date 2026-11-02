"""Program to find low-rank tensor decomposition for 3x3x7 matrix multiplication."""
DIMENSIONS = [3, 3, 7]

import time
import random
from fractions import Fraction
import numpy as np

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    num_U = len(U)
    num_V = len(V)
    num_W = len(W)
    if num_U != n * m:
        return False
    if num_V != m * p:
        return False
    if num_W != p * n:
        return False
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The rank-1 term: U[i*m+j,c] * V[j*p+k,c] should equal W[k*n+i,c]
                    # for the matrix multiplication tensor
                    expected = U[i * m + j][c] * V[j * p + k][c]
                    actual = W[k * n + i][c]
                    if expected != actual:
                        return False
    return True

def _try_find_decomposition(rank, n, m, p, max_iters=100, seed=42):
    """
    Try to find a decomposition of given rank.
    Uses random U, V and solves linear system for W.
    """
    rng = np.random.RandomState(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Equations: 63
    # Variables: W_size * rank
    num_eqs = n * m * p
    num_vars = W_size * rank
    
    # We will try multiple random initializations of U and V
    for iter_idx in range(max_iters):
        # Generate random U and V with small integer/half-integer values
        # Using values from {-1, -0.5, 0, 0.5, 1} helps with integer solutions
        vals = np.array([-1, -0.5, 0, 0.5, 1])
        U = rng.choice(vals, size=(U_size, rank))
        V = rng.choice(vals, size=(V_size, rank))
        
        # Build A and b
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # W index is k*n + i
                    w_idx = k * n + i
                    # For each c, the coefficient of W[w_idx, c] is U[i*m+j, c] * V[j*p+k, c]
                    for c in range(rank):
                        A[eq_idx, w_idx * rank + c] = U[i * m + j, c] * V[j * p + k, c]
                    eq_idx += 1
        
        # Solve A w = b
        # Since num_vars > num_eqs, we use least squares to find a solution
        # We prefer solutions with small values, so minimum norm is a good start
        try:
            w, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is close to half-integers
        # Snap to half-integers
        w_snapped = np.round(w * 2) / 2.0
        
        # Check residual with snapped values
        w_res = A @ w_snapped - b
        if np.max(np.abs(w_res)) < 1e-9:
            # We found a solution!
            # Reshape w_snapped back to W
            W = w_snapped.reshape(W_size, rank)
            
            # Convert to lists of floats/ints for the return format
            # Ensure they are exactly representable (0.5, 1.0, etc.)
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify exactness with the original definition
            if _verify_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not exact, we could try to adjust, but for now, just try next random init
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound for 3x3x3 is 20. Assume 20 for 3x3x7.
    # If 20 is too low, we will just move up.
    
    ranks_to_try = list(range(20, best_rank))
    
    base_seed = 42
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Limit iterations per rank to avoid getting stuck
        max_iters = 100
        result = _try_find_decomposition(r, n, m, p, max_iters=max_iters, seed=base_seed + r)
        
        if result is not None:
            best = result
            best_rank = r
            # Since we are iterating from low to high, the first found is the best.
            return best
            
    return best
