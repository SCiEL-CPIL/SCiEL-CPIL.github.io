"""Find rank-minimal exact tensor decomposition for 2x3 x 3x5 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Search for the lowest possible rank
    # Known lower bounds suggest rank can be significantly lower than 30
    # Try ranks from 20 to 29
    for r in range(20, 30):
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook decomposition (rank 30)
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

def _find_decomposition(n, m, p, r, deadline):
    nm, mp, pn = n*m, m*p, p*n
    
    # Number of trials for this rank
    max_trials = 30
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
            
        # Random initialization
        # Use a mix of random and structured initialization
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
        
        # Alternating Least Squares (ALS)
        converged = False
        for it in range(200):
            if time.time() > deadline:
                return None
                
            # Update U
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    A = np.zeros(r)
                    for k in range(p):
                        A += V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.norm(A) > 1e-10:
                        # Solve min || A^T u - 1 ||
                        U_new[i*m+j, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    A = np.zeros(r)
                    for i in range(n):
                        A += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(A) > 1e-10:
                        V_new[j*p+k, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
            
            # Update W
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
            
            if diff < 1e-12:
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                    converged = True
                    break
        
        if converged:
            # Polish the solution to higher precision
            U, V, W = _polish(U, V, W, n, m, p, deadline)
            
            # Rationalize with larger denominator limit
            U_rat = _rationalize(U, max_denom=1000)
            V_rat = _rationalize(V, max_denom=1000)
            W_rat = _rationalize(W, max_denom=1000)
            
            if U_rat is not None and V_rat is not None and W_rat is not None:
                if _verify_decomposition_exact(U_rat, V_rat, W_rat, n, m, p):
                    return {"U": U_rat, "V": V_rat, "W": W_rat}
            
            # If rationalization with 1000 fails, try 10000 (slower but more robust)
            if time.time() < deadline - 10:
                U_rat = _rationalize(U, max_denom=10000)
                V_rat = _rationalize(V, max_denom=10000)
                W_rat = _rationalize(W, max_denom=10000)
                if U_rat is not None and V_rat is not None and W_rat is not None:
                    if _verify_decomposition_exact(U_rat, V_rat, W_rat, n, m, p):
                        return {"U": U_rat, "V": V_rat, "W": W_rat}
    
    return None

def _polish(U, V, W, n, m, p, deadline):
    """Run additional ALS iterations to tighten the solution."""
    for it in range(100):
        if time.time() > deadline:
            break
            
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                A = np.zeros(len(U[0]))
                for k in range(p):
                    A += V[j*p+k, :] * W[k*n+i, :]
                if np.linalg.norm(A) > 1e-10:
                    U_new[i*m+j, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
        
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                A = np.zeros(len(V[0]))
                for i in range(n):
                    A += U[i*m+j, :] * W[k*n+i, :]
                if np.linalg.norm(A) > 1e-10:
                    V_new[j*p+k, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
        
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                A = np.zeros(len(W[0]))
                for j in range(m):
                    A += U[i*m+j, :] * V[j*p+k, :]
                if np.linalg.norm(A) > 1e-10:
                    W_new[k*n+i, :] = np.linalg.lstsq(A.reshape(1,-1), np.ones(1), rcond=None)[0].flatten()
        
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if diff < 1e-14:
            break
            
    return U, V, W

def _rationalize(arr, max_denom=1000):
    """Convert float array to Fraction array if close to rationals."""
    result = []
    for row in arr:
        row_rat = []
        for val in row:
            if abs(val) < 1e-10:
                row_rat.append(Fraction(0))
                continue
                
            best_frac = None
            best_err = 1e-8
            
            # Try denominators 1 to max_denom
            for q in range(1, max_denom + 1):
                p = round(val * q)
                err = abs(val - p/q)
                if err < best_err:
                    best_err = err
                    best_frac = Fraction(p, q)
                if err < 1e-12:
                    break
            
            if best_err > 1e-8:
                return None
            row_rat.append(best_frac)
        result.append(row_rat)
    return result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition with floating point arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify decomposition with exact Fraction arithmetic."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                if val != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
