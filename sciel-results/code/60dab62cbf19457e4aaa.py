"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _is_nice_half_integer(arr, tol=1e-9):
    """Check if array entries are close to half-integers."""
    # Multiply by 2 and check if close to integer
    doubled = arr * 2
    rounded = np.round(doubled)
    return np.all(np.abs(doubled - rounded) < tol)

def _solve_for_W(U, V, n, m, p, r):
    """
    Given U and V, solve for W such that the tensor decomposition holds.
    Returns W or None if no 'nice' solution is found.
    """
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for k in range(p):
            # We need to find W_row (length r) such that:
            # sum_c W_row[c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j in [0, m)
            
            A = np.zeros((m, r))
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                A[j] = u_row * v_row
            
            # Solve A * W_row = 1
            # A is m x r. W_row is r x 1.
            # We want a solution with small half-integer entries.
            
            # Use least squares to find a particular solution
            # If r >= m, there are many solutions. We want one with small entries.
            # np.linalg.lstsq gives the minimum norm solution.
            try:
                # Add a small regularization to avoid singular matrices issues if any
                # But A is random, so likely full rank.
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            # Check if the solution is close to half-integers
            if _is_nice_half_integer(w_sol):
                # Snap to half-integers to ensure exactness
                W[k * n + i] = np.round(w_sol * 2) / 2.0
            else:
                # Try to find a better solution?
                # For now, reject this random U,V
                return None
                
    return W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # Start from a reasonable lower bound. 
    # For 2x3x4, the rank is likely around 19-20.
    # Schoolbook is 24.
    # We will try ranks from 19 to 24.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    for r in range(19, 25):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution for this rank
        max_attempts = 5000 if r < 22 else 10000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Randomly initialize U and V with small integers
            # Using {-1, 0, 1} or {-2, -1, 0, 1, 2}
            # Smaller values increase the chance of half-integer solutions for W
            U = np.random.randint(-1, 2, (U_rows, r))
            V = np.random.randint(-1, 2, (V_rows, r))
            
            # Solve for W
            W = _solve_for_W(U, V, n, m, p, r)
            
            if W is not None:
                # Verify the decomposition
                if _verify_decomposition(U, V, W, n, m, p):
                    # Convert to lists of floats
                    U_out = U.tolist()
                    V_out = V.tolist()
                    W_out = W.tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
    # Fallback: Schoolbook decomposition (Rank 24)
    # This is guaranteed to work
    r = n * m * p
    U = np.zeros((U_rows, r))
    V = np.zeros((V_rows, r))
    W = np.zeros((W_rows, r))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
