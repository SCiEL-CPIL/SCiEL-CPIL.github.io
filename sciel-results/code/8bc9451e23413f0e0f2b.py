"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), the product should equal 1 if it's a valid term, 0 otherwise.
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # in the sense that the multiplication tensor is sum_{i,j,k} e_i^T ⊗ e_j^T ⊗ e_k^T
    # No wait: the multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This must hold for all i in [0,n), j in [0,m), k in [0,p).
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_search(max_rank=35, deadline=None):
    """Try to find a decomposition of rank <= max_rank using alternating minimization."""
    n, m, p = DIMENSIONS
    best = None
    for r in range(15, max_rank + 1):
        if deadline and time.time() > deadline - 5:
            break
        # Try multiple random initializations
        for attempt in range(20):
            if deadline and time.time() > deadline - 5:
                break
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            # Scale to match the tensor
            for _ in range(50):
                # Update U
                for i in range(n):
                    for j in range(m):
                        U[i*m+j] = np.sum(V[j*p:] * W[:n] * np.eye(p)*0, axis=0)  # placeholder
                # This is getting complex; let's use a simpler approach
            # Skip complex optimization for now, try structured approach
        # Try structured approach: use known 2x3x3 decomposition doubled
    return best

def _structured_decomposition():
    """
    Try to construct a lower-rank decomposition using block structure.
    For 2x3x6, we can view the 6 columns as two blocks of 3.
    If we have a rank-r decomposition for 2x3x3, we can potentially share
    the A-side factors.
    
    Actually, let's try: C = A*B where A is 2x3, B is 3x6.
    Write B = [B1 | B2] where B1, B2 are 3x3.
    Then C = [A*B1 | A*B2].
    
    If we use the same set of intermediate values for both A*B1 and A*B2,
    we can save rank. Specifically, if the rank-r decomposition of 2x3x3
    uses factors U, V, W, then for 2x3x6 we need:
    - U stays the same (n*m x r)
    - V becomes [V1; V2] where V1, V2 are (m*p/2) x r each? No...
    
    Let me think differently. The multiplication tensor for 2x3x6:
    T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,6)
    
    A rank-r decomposition: T[i,j,k] = sum_c U[i*3+j,c] * V[j*6+k,c] * W[k*2+i,c]
    
    One idea: use the fact that the k-index has 6 values. We can try to
    express the 6-dimensional output space using fewer than 6 independent
    "directions" in the W factor.
    
    Let's try a simple structured approach: assume W has a block structure
    W[k*2+i, c] where for fixed i, the 6 values across k can be represented
    by a lower-dimensional code.
    """
    n, m, p = 2, 3, 6
    # Start with schoolbook and try to find linear dependencies
    # For now, return schoolbook as fallback
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

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    # Try to find a better decomposition
    result = _structured_decomposition()
    return result
# EVOLVE-BLOCK-END
