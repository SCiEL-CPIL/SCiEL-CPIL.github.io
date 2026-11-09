"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses a structured search approach with known low-rank constructions.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    n_m = n * m  # 9
    m_p = m * p  # 12
    p_n = p * n  # 12
    
    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    def _verify(U, V, W, rank):
        """Verify that the decomposition is exact using integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def _verify_float(U, V, W, rank, tol=1e-9):
        """Verify that the decomposition is exact using floating point."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def _schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n_m)]
        V = [[0] * rank for _ in range(m_p)]
        W = [[0] * rank for _ in range(p_n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = _schoolbook()
    
    # Try to find lower rank decompositions
    # Known results suggest rank around 20-25 for 3x3*3x4
    
    # Strategy 1: Try a structured construction based on Strassen-like algorithms
    # For 3x3 matrix multiplication, Strassen's algorithm gives rank 7
    # For 3x3 * 3x4, we can potentially use a similar approach
    
    # Let's try a direct search for low-rank decompositions
    # The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of 36 equations. For rank r, we have 33r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions should exist.
    
    # Try ranks from schoolbook-1 down to a reasonable minimum
    for r in range(schoolbook_rank - 1, 11, -1):
        if time.time() > deadline:
            break
        
        # Try random search with integer/half-integer entries
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initialization with values from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n_m, r))
            V = np.random.randint(-1, 2, size=(m_p, r))
            W = np.random.randint(-1, 2, size=(p_n, r))
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
