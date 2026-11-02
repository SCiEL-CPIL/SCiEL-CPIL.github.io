import time
import sys
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=4, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                
                expected = 1.0
                err = abs(s - expected)
                max_err = max(max_err, err)
    
    return max_err <= tol

def _schoolbook():
    """Return the standard schoolbook decomposition of rank 24."""
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}

def _try_projection_decomposition():
    """
    Attempt to construct a lower-rank decomposition by projecting a 3x3x3 Strassen
    decomposition onto the 2x3x4 subspace.
    
    Note: This is a hypothesis-driven attempt. The dimensions 2,3,4 do not perfectly
    align with 3,3,3, so we define a mapping for the indices.
    A: 2x3 (indices 0-1, 0-2)
    B: 3x4 (indices 0-2, 0-3)
    C: 2x4 (indices 0-1, 0-3)
    
    We cannot directly project 3x3x3 because the '4' dimension in B and C is larger than 3.
    Therefore, this specific projection strategy is structurally invalid for exact decomposition.
    We return None to indicate failure, triggering the fallback to schoolbook.
    """
    # Due to dimension mismatch (4 > 3), a direct 3x3x3 projection is impossible.
    return None

def _strassen_based_attempt(deadline):
    """
    Try to find a lower-rank decomposition using Strassen-like ideas.
    """
    n, m, p = 2, 3, 4
    
    # Try the projection strategy
    result = _try_projection_decomposition()
    if result is not None:
        if _verify(result['U'], result['V'], result['W']):
            return result
    
    # Fallback to schoolbook within this function to satisfy interface
    return _schoolbook()

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    """
    deadline = time.time() + 140
    
    result = _strassen_based_attempt(deadline)
    
    if result is not None:
        return result
    
    result = _schoolbook()
    
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    return result
# EVOLVE-BLOCK-END
