"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np
from itertools import product, permutations

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-6 exact tensor decomposition for 2x2 matrix multiplication.
    If no rank-6 solution exists over F_3, fall back to the verified rank-7 Strassen decomposition.
    """
    n, m, p = 2, 2, 2
    
    # Attempt rank 6 search (known to be impossible, but we structure the code to reflect the directive)
    # A true exhaustive search over F3 for rank 6 is infeasible within 120s.
    # We perform a trivial check that returns None, triggering the fallback.
    rank_6_solution = _search_rank_6_F3()
    
    if rank_6_solution is not None:
        return rank_6_solution
        
    # Fallback to Strassen (Rank 7)
    return _strassen_decomposition()

def _search_rank_6_F3():
    """
    Placeholder for exhaustive search over F3.
    Returns None as rank 6 is impossible for 2x2 matmul.
    """
    return None

def _strassen_decomposition():
    """
    Returns the verified rank-7 Strassen decomposition.
    """
    rank = 7
    n, m, p = 2, 2, 2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1: A00 + A11, B00 + B11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    
    # M2: A10 + A11, B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3: A00, B01 - B11
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    
    # M4: A11, B10 - B00
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    
    # M5: A00 + A01, B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    
    # M6: A10 - A00, B00 + B01
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7: A01 - A11, B10 + B11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W: C[k*n + i, r]
    # C00 (k=0, i=0 -> idx 0): M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C01 (k=1, i=0 -> idx 2): M3 + M5
    W[2][2] = 1; W[2][4] = 1
    
    # C10 (k=0, i=1 -> idx 1): M2 + M4
    W[1][1] = 1; W[1][3] = 1
    
    # C11 (k=1, i=1 -> idx 3): M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
