"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
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


def _is_half_integer(x, tol=1e-9):
    # Check if x is close to a half-integer
    k = round(2 * x)
    return abs(2 * x - k) < tol


def _snap_to_half_integer(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return x


def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(u).limit_denominator(10**6) for u in row] for row in U]
    Vf = [[Fraction(v).limit_denominator(10**6) for v in row] for row in V]
    Wf = [[Fraction(w).limit_denominator(10**6) for w in row] for row in W]
    
    # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has T[i,j,k] = 1 if we're multiplying, 0 otherwise
    # Actually, the tensor for matrix multiplication is T[i, j, k] = 1 for all i,j,k in the appropriate ranges
    # Wait, let me reconsider. The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1
    # where i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for c in range(rank):
                    actual += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if actual != expected:
                    return False
    return True


def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, deadline=None):
    """ALS (Alternating Least Squares) for tensor decomposition of matrix multiplication."""
    # Initialize randomly
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then M @ U[i*m+j, :].T = 1 (vector of ones of length p)
        
        for i in range(n):
            for j in range(m):
                # Build M: p x rank
                M = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Solve M @ x = ones(p)
                try:
                    x, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), V[j*p+k, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        # Let M[i,c] = U[i*m+j,c] * W[k*n+i,c]
        # M @ V[j*p+k, :].T = 1 (vector of ones of length n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    for c in range(range(rank)):
                        M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    x, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), W[k*n+i, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        # Let M[j,c] = U[i*m+j,c] * V[j*p+k,c]
        # M @ W[k*n+i, :].T = 1 (vector of ones of length m)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    x, residuals, rank_M, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence: compute the error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W


def _snap_and_verify(U, V, W):
    """Snap to half-integers and verify."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_snapped = [[_snap_to_half_integer(U[i][c]) for c in range(rank)] for i in range(n*m)]
    V_snapped = [[_snap_to_half_integer(V[i][c]) for c in range(rank)] for i in range(m*p)]
    W_snapped = [[_snap_to_half_integer(W[i][c]) for c in range(rank)] for i in range(p*n)]
    
    # Check if all are half-integers
    for row in U_snapped:
        for x in row:
            if not _is_half_integer(x, tol=1e-6):
                return None
    for row in V_snapped:
        for x in row:
            if not _is_half_integer(x, tol=1e-6):
                return None
    for row in W_snapped:
        for x in row:
            if not _is_half_integer(x, tol=1e-6):
                return None
    
    # Verify
    if _verify(U_snapped, V_snapped, W_snapped):
        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Also rank >= ceil(np.sqrt(n*m*p)) and other bounds
    # For 3x4x4, let's try from 16 up to 48
    
    best_result = None
    
    for rank in range(16, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            try:
                U, V, W = _als_decompose(n, m, p, rank, max_iter=100, tol=1e-12, deadline=deadline - 5)
            except:
                continue
            
            result = _snap_and_verify(U, V, W)
            if result is not None:
                return result
    
    # If we couldn't find a better decomposition, return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
