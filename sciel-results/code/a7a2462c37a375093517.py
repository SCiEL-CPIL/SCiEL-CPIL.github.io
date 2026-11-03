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


def _is_rational(x, tol=1e-8, max_den=100):
    """Check if x is close to a rational with small denominator."""
    for d in range(1, max_den + 1):
        n = round(x * d)
        if abs(x * d - n) < tol:
            return True
    return False


def _snap_to_rational(x, tol=1e-6, max_den=100):
    """Snap x to the nearest rational with small denominator."""
    best_frac = x
    best_dist = tol
    for d in range(1, max_den + 1):
        n = round(x * d)
        dist = abs(x * d - n) / d
        if dist < best_dist:
            best_dist = dist
            best_frac = n / d
    return best_frac


def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(u).limit_denominator(10**6) for u in row] for row in U]
    Vf = [[Fraction(v).limit_denominator(10**6) for v in row] for row in V]
    Wf = [[Fraction(w).limit_denominator(10**6) for w in row] for row in W]
    
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


def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, deadline=None, seed=42):
    """ALS (Alternating Least Squares) for tensor decomposition of matrix multiplication."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j)
        for i in range(n):
            for j in range(m):
                # M[k,c] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k)
        for j in range(m):
            for k in range(p):
                # M[i,c] = U[i*m+j, c] * W[k*n+i, c]
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i)
        for k in range(p):
            for i in range(n):
                # M[j,c] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W


def _snap_and_verify(U, V, W, max_den=50):
    """Snap to rationals with small denominators and verify."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_snapped = [[_snap_to_rational(U[i][c], max_den=max_den) for c in range(rank)] for i in range(n*m)]
    V_snapped = [[_snap_to_rational(V[i][c], max_den=max_den) for c in range(rank)] for i in range(m*p)]
    W_snapped = [[_snap_to_rational(W[i][c], max_den=max_den) for c in range(rank)] for i in range(p*n)]
    
    # Check if all are close to rationals with small denominators
    for row in U_snapped:
        for x in row:
            if not _is_rational(x, max_den=max_den):
                return None
    for row in V_snapped:
        for x in row:
            if not _is_rational(x, max_den=max_den):
                return None
    for row in W_snapped:
        for x in row:
            if not _is_rational(x, max_den=max_den):
                return None
    
    # Verify
    if _verify(U_snapped, V_snapped, W_snapped):
        return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 16
    for rank in range(16, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            try:
                U, V, W = _als_decompose(n, m, p, rank, max_iter=50, tol=1e-10, 
                                         deadline=deadline - 5, seed=seed)
            except:
                continue
            
            # Try snapping with different max denominators
            for max_den in [1, 2, 4, 8, 16, 32, 50]:
                if time.time() > deadline - 5:
                    break
                result = _snap_and_verify(U, V, W, max_den=max_den)
                if result is not None:
                    return result
    
    # If we couldn't find a better decomposition, return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
