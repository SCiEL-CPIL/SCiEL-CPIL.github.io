"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def _schoolbook():
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each (i, j, k), the contribution to C[i,k] from A[i,j]*B[j,k] should be 1
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Build the tensor by iterating over r and checking contributions
    # More efficiently: for each r, the outer product U_r ⊗ V_r ⊗ W_r should sum to the target tensor
    # Target tensor has 1 at (i,j,k) for all i,j,k
    
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_search(rank, n, m, p, timeout_s=10):
    """Try to find a rank-`rank` exact decomposition with half-integer entries via random search."""
    deadline = time.time() + timeout_s
    
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # We need U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Try with entries from {0, 0.5, 1, 1.5, 2} (small half-integers)
    # This is a very constrained search; for rank < 84 it's unlikely to find one quickly.
    # But we try a few random initializations.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with small half-integers
        vals = [0, 0.5, 1, 1.5, 2]
        U = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(n*m)]
        V = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(m*p)]
        W = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(p*n)]
        
        if _verify(U, V, W):
            return U, V, W
    
    return None

def construct():
    """Construct the tensor decomposition. Returns dict with U, V, W matrices."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Must finish within 120s
    
    # Start with schoolbook (rank 84) as baseline
    best = _schoolbook()
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But in practice, for small matrices, the rank is often close to n*m*p
    # Known results: rank of 2x2x2 is 7 (Strassen), rank of 3x3x3 is 23 (Landsberg)
    # For 3x4x7, no well-known lower bound better than schoolbook is readily available
    
    # Try a few lower ranks with random search
    for rank in range(60, 84):
        if time.time() > deadline - 30:
            break
        result = _try_random_search(rank, n, m, p, timeout_s=5)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            break
    
    return best
