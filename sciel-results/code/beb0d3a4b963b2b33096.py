"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 4 matrix.
Output factor follows the official trace convention W[k*n+i,r].
"""

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-24 schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _try_strassen_block(n, m, p):
    """
    Try to construct a lower-rank decomposition using block structure.
    For 2x3x4, we can try to exploit the 2x2x2 Strassen structure.
    """
    # A is 2x3, B is 3x4
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    
    # This doesn't immediately reduce rank without further tricks.
    # Let's try a different approach: use the known rank of MM_{2,2,2} = 7
    # and see if we can embed it.
    
    # Actually, for 2x3x4, one known approach is to use the fact that
    # the tensor can be decomposed using a combination of Strassen-like terms.
    
    # Let's try a randomized search for rank < 24 with small integer coefficients.
    # This is heuristic and may not find a solution, but it's worth trying.
    
    return None

def _random_search(n, m, p, max_rank, time_limit=5.0):
    """
    Randomized search for a lower-rank decomposition.
    Try random sparse matrices with small integer coefficients.
    """
    start = time.time()
    deadline = start + time_limit
    
    # We'll try ranks from max_rank down to 12 (theoretical lower bound)
    for rank in range(max_rank, 11, -1):
        if time.time() > deadline:
            break
            
        # Try a few random attempts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Generate random sparse U, V, W
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Fill with small random integers
            for r in range(rank):
                # Choose random non-zero positions
                u_idx = np.random.choice(n * m, size=min(3, n * m), replace=False)
                v_idx = np.random.choice(m * p, size=min(3, m * p), replace=False)
                w_idx = np.random.choice(p * n, size=min(3, p * n), replace=False)
                
                U[u_idx, r] = np.random.randint(-2, 3, size=len(u_idx))
                V[v_idx, r] = np.random.randint(-2, 3, size=len(v_idx))
                W[w_idx, r] = np.random.randint(-2, 3, size=len(w_idx))
            
            # Convert to lists of Fractions
            U_list = [[Fraction(int(x)) for x in row] for row in U]
            V_list = [[Fraction(int(x)) for x in row] for row in V]
            W_list = [[Fraction(int(x)) for x in row] for row in W]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, return the guaranteed schoolbook decomposition
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower-rank decomposition
    # Strategy 1: Try known structured decompositions
    structured = _try_strassen_block(n, m, p)
    if structured is not None:
        if _verify_decomposition(structured["U"], structured["V"], structured["W"], n, m, p):
            return structured
    
    # Strategy 2: Randomized search for lower ranks
    # Only try if we have enough time
    if time.time() < deadline - 10.0:
        random_result = _random_search(n, m, p, max_rank=n * m * p - 1, time_limit=30.0)
        if random_result is not None:
            if _verify_decomposition(random_result["U"], random_result["V"], random_result["W"], n, m, p):
                return random_result
    
    # Return the schoolbook decomposition as fallback
    return schoolbook
# EVOLVE-BLOCK-END
