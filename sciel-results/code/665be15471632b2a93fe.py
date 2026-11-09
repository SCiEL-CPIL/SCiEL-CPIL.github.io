"""Find rank-minimal exact tensor decomposition for 2x3 x 3x5 matrix multiplication."""
import numpy as np
from itertools import product
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try to find a decomposition of rank r < 30
    # Use a randomized approach with exact arithmetic
    
    best_rank = nm * mp  # schoolbook fallback: 30
    
    # Try various ranks
    for r in range(24, 30):
        if _find_decomposition(n, m, p, r):
            best_rank = r
            break
    
    if best_rank < 30:
        # Find the actual decomposition
        U, V, W = _find_decomposition(n, m, p, best_rank)
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
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

def _find_decomposition(n, m, p, r, max_iters=50):
    """Try to find a rank-r decomposition using randomized initialization and Newton's method."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the list of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for trial in range(max_iters):
        # Random initialization
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Normalize columns to avoid scaling issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) + np.sum(V[:,c]**2) + np.sum(W[:,c]**2))
            if norm > 1e-10:
                U[:,c] /= norm
                V[:,c] /= norm
                W[:,c] /= norm
        
        # Use alternating least squares / fixed point iteration
        converged = False
        for it in range(200):
            # Update U: for each (i,j), solve sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # For each k, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is a system in U[i*m+j,:]
                    # A[c] = sum_k V[j*p+k,c] * W[k*n+i,c]
                    # Then sum_c U[i*m+j,c] * A[c] = 1
                    A = np.zeros(r)
                    for k in range(p):
                        A += V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j,:] * A = 1 (least squares)
                    if np.linalg.norm(A) > 1e-10:
                        U_new[i*m+j, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
            
            # Update V: for each (j,k), solve sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A = np.zeros(r)
                    for i in range(n):
                        A += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(A) > 1e-10:
                        V_new[j*p+k, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
            
            # Update W: for each (k,i), solve sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    A = np.zeros(r)
                    for j in range(m):
                        A += U[i*m+j, :] * V[j*p+k, :]
                    if np.linalg.norm(A) > 1e-10:
                        W_new[k*n+i, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
            
            # Check convergence
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < 1e-10:
                # Verify the solution
                if _verify_decomposition(U, V, W, n, m, p):
                    converged = True
                    break
        
        if converged:
            # Try to rationalize
            U_rat = _rationalize(U)
            V_rat = _rationalize(V)
            W_rat = _rationalize(W)
            if U_rat is not None and V_rat is not None and W_rat is not None:
                if _verify_decomposition_exact(U_rat, V_rat, W_rat, n, m, p):
                    return [U_rat.tolist(), V_rat.tolist(), W_rat.tolist()]
            else:
                # Use floating point if rationalization fails
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact Fraction arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c]) for c in range(len(U[0])))
                if val != 1:
                    return False
    return True

def _rationalize(arr, max_denom=100):
    """Try to convert a numpy array to a list of Fractions."""
    result = []
    for row in arr:
        row_rat = []
        for val in row:
            frac = Fraction(val).limit_denominator(max_denom)
            if abs(float(frac) - val) > 1e-8:
                return None
            row_rat.append(frac)
        result.append(row_rat)
    return result

# EVOLVE-BLOCK-END
