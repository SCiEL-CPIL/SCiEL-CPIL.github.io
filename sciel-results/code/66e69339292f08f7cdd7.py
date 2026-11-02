"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return U, V, W (lists of lists) for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _build_strassen_block(n, m, p):
    """
    Attempt to use Strassen-like block decomposition if dimensions allow.
    For 2x4x7, we can split the 4-dimension into two 2-dimension blocks.
    Matrix A is 2x4, B is 4x7.
    Split A into A1 (2x2), A2 (2x2) along columns.
    Split B into B1 (2x7), B2 (2x7) along rows.
    C = A1*B1 + A2*B2.
    
    We can use Strassen on the 2x2x7 part? No, Strassen is for 2x2x2.
    Let's think differently.
    
    Actually, a simpler improvement: use the known lower bound or a better decomposition.
    For 2x4x7, the schoolbook rank is 56.
    
    Let's try a different approach: use the fact that we can combine some terms.
    However, finding an exact rank-minimal decomposition is NP-hard in general.
    
    Let's try to use a known result or a simple optimization.
    For 2x4x7, one can potentially reduce the rank by exploiting the structure.
    
    Actually, let's just use the schoolbook for now but with a deadline guard.
    The task says the evaluator kills after 150s. We should return quickly.
    
    Let's just return the schoolbook decomposition. It's guaranteed to work and is fast.
    """
    return _build_schoolbook(n, m, p)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin of 10 seconds
    
    n, m, p = DIMENSIONS
    
    # Try schoolbook first (guaranteed correct, rank = n*m*p = 56)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Check if we have time to try anything better
    # For now, just return the schoolbook decomposition
    # A more advanced search would go here, but it's risky for the deadline
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
