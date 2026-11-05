"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _matmul_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    T = _matmul_tensor(n, m, p)
    for c in range(rank):
        # Compute outer product of U[:,c], V[:,c], W[:,c]
        u = np.array(U[:, c], dtype=np.int64) if hasattr(U, 'dtype') else np.array([row[c] for row in U], dtype=np.int64)
        v = np.array(V[:, c], dtype=np.int64) if hasattr(V, 'dtype') else np.array([row[c] for row in V], dtype=np.int64)
        w = np.array(W[:, c], dtype=np.int64) if hasattr(W, 'dtype') else np.array([row[c] for row in W], dtype=np.int64)
        # The contribution is u[i] * v[j] * w[k] for each (i,j,k)
        # Sum over c should equal T
    # More efficient: reconstruct
    # Actually, let's do it properly: the tensor is sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We need to check this equals T
    # For efficiency with small dimensions, do it directly
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for c in range(rank):
        uc = np.array(U[:, c], dtype=np.int64) if hasattr(U, 'dtype') else np.array([row[c] for row in U], dtype=np.int64)
        vc = np.array(V[:, c], dtype=np.int64) if hasattr(V, 'dtype') else np.array([row[c] for row in V], dtype=np.int64)
        wc = np.array(W[:, c], dtype=np.int64) if hasattr(W, 'dtype') else np.array([row[c] for row in W], dtype=np.int64)
        # Outer product
        T_recon += np.einsum('i,j,k->ijk', uc, vc, wc)
    return np.array_equal(T_recon, T)

def _try_strassen_2x2(n, m, p):
    """
    Try to use Strassen's algorithm on 2x2 blocks along the middle dimension.
    This works when m is even. For m=3, we can't directly split into 2x2 blocks.
    But we can try a different approach.
    """
    # For m=3, we can't use 2x2 Strassen directly on the middle dimension.
    # Let's return None to indicate this strategy doesn't apply.
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_polynomial_interpolation(n, m, p):
    """
    Try a polynomial interpolation approach on the output dimension.
    The idea: represent the p output columns as evaluations of a degree-(p-1) polynomial.
    If we can find a basis transformation that makes the tensor have a lower rank,
    we win.
    
    For multiplying A (n x m) by B (m x p), the output C (n x p) has:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we think of each row of C as a function of k, then C[i,k] is a "polynomial" 
    in some sense. But this is a stretch for general matrices.
    
    Let's try a different angle: use the fact that for small n=2, m=3, p=10,
    we can try to find a decomposition with rank < 60 by exploiting structure.
    """
    # This is a placeholder; the actual strategy needs more thought.
    # For now, return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-like approaches
    if m % 2 == 0:
        result = _try_strassen_2x2(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                rank = len(U[0]) if U else 0
                if rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
    
    # Try polynomial interpolation
    result = _try_polynomial_interpolation(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            rank = len(U[0]) if U else 0
            if rank < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
    
    # For m=3, try a hybrid approach: split m=3 into 1+2 and use Strassen on the 2x2 part
    # This is speculative; let's just return the best we have.
    
    return best
# EVOLVE-BLOCK-END
