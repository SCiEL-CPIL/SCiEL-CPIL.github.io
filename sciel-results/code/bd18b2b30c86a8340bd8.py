"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i1 in range(n):
        for r in range(m):
            for j2 in range(p):
                i = i1 * m + r
                j = r * p + j2
                k = i1 * n + j2
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _is_half_integer(mat, tol=1e-6):
    return np.allclose(2 * mat, np.round(2 * mat), atol=tol)

def _build_rank18():
    """Construct the known rank-18 decomposition (two Strassen blocks + schoolbook for col 4)."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _gaussian_eliminate_rational(A, b, max_pivot_search=100):
    """
    Solve A x = b using Gaussian elimination over rationals.
    A is a list of lists of Fractions, b is a list of Fractions.
    Returns x as a list of Fractions, or None if no unique solution.
    """
    n = len(b)
    if len(A) != n:
        return None
    m = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for col in range(min(n, m)):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Normalize pivot row
        pivot_val = M[col][col]
        for j in range(col, m + 1):
            M[col][j] /= pivot_val
        
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            if factor != 0:
                for j in range(col, m + 1):
                    M[row][j] -= factor * M[col][j]
    
    # Back substitution / extract solution
    x = [Fraction(0)] * m
    for i in range(min(n, m)):
        # Check if row i has a pivot at column i
        if M[i][i] != 0:
            x[i] = M[i][m]
        else:
            # Free variable or inconsistent
            if M[i][m] != 0:
                return None  # Inconsistent
            # Free variable, set to 0 for now
    
    # Verify solution
    for i in range(n):
        s = sum(A[i][j] * x[j] for j in range(m))
        if s != b[i]:
            return None
    
    return x

def _try_rank17_algebraic():
    """
    Try to find a rank-17 decomposition by starting from rank-18 and attempting to
    merge two terms or find a different structure.
    
    Strategy: The rank-18 solution has a block-diagonal structure (two independent
    Strassen blocks for cols 0-1 and 2-3, plus schoolbook for col 4).
    
    To get rank 17, we need cross-term cancellations. We'll try:
    1. Start with rank-18 solution
    2. Try to express one term as a linear combination of others
    3. Use rational Gaussian elimination to solve for modified entries
    
    Alternative: Directly set up the linear system for rank 17 with sparse structure.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    
    # The tensor T has 20 non-zero entries (one for each (i1, r, j2) combination)
    # We need: T[i,j,k] = sum_{r=0}^{R-1} U[i,r] * V[j,r] * W[k,r]
    
    # For rank 17, we have 17 * (4 + 10 + 10) = 340 unknowns
    # And 20 equations (from non-zero T entries) + constraints from zero entries
    
    # Actually, we have 4*10*10 = 400 total entries, but only 20 are non-zero.
    # So we have 20 equations from non-zeros and 380 from zeros.
    # Total: 400 equations, 340 unknowns. This is overdetermined.
    
    # Let's try a different approach: use the structure of the problem.
    # The matrix multiplication tensor for 2x2 * 2x5 can be seen as:
    # For each output column j2 (0..4), we need to compute:
    #   C[i1, j2] = sum_{r=0}^{1} A[i1, r] * B[r, j2]
    
    # This is like 5 independent 2x2 * 2x1 multiplications, but they share the A matrix.
    
    # Let's try to find a rank-17 solution by perturbing the rank-18 solution
    # and solving the resulting linear system.
    
    U18, V18, W18 = _build_rank18()
    
    # Try to find a rank-17 solution by dropping one column and adjusting others.
    # We'll try dropping each of the 18 columns and solving for the remaining 17.
    
    for drop_col in range(18):
        # Build the system: for each non-zero T[i,j,k], 
        # T[i,j,k] = sum_{r != drop_col} U[i,r] * V[j,r] * W[k,r]
        
        # This is still nonlinear (product of three unknowns).
        # We need a different approach.
        
        # Let's fix V and W to be the rank-18 values (excluding drop_col),
        # and solve for U.
        # Then check if the reconstruction is correct.
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Copy all columns except drop_col
        for r in range(R):
            orig_r = r if r < drop_col else r + 1
            U[:, r] = U18[:, orig_r]
            V[:, r] = V18[:, orig_r]
            W[:, r] = W18[:, orig_r]
        
        # Now solve for U given V and W
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For each i: T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])_outer
        
        n, m, p = DIMENSIONS
        solved_U = np.zeros((n*m, R))
        for i in range(n*m):
            # Build linear system for U[i,:]
            # b = T[i,:,:].flatten()
            # A[idx, r] = V[j,r] * W[k,r] where idx = j*(p*n) + k
            b = T[i].flatten()
            A = np.zeros((m*p * p*n, R))
            for r in range(R):
                outer = np.outer(V[:, r], W[:, r])
                A[:, r] = outer.flatten()
            
            # Solve least squares
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                solved_U[i] = x
            except:
                break
        
        U = solved_U
        
        # Check reconstruction
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        
        if err < 1e-6:
            # Found a solution! Check if it's half-integer
            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                if _check_decomposition(U, V, W, T):
                    return U, V, W
        
        # Try also solving for V given U and W
        solved_V = np.zeros((m*p, R))
        for j in range(m*p):
            b = T[:, j, :].transpose(1, 0).flatten()
            A = np.zeros((n*m * p*n, R))
            for r in range(R):
                outer = np.outer(U[:, r], W[:, r])
                A[:, r] = outer.flatten()
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                solved_V[j] = x
            except:
                break
        
        V = solved_V
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                if _check_decomposition(U, V, W, T):
                    return U, V, W
        
        # Try solving for W given U and V
        solved_W = np.zeros((p*n, R))
        for k in range(p*n):
            b = T[:, :, k].flatten()
            A = np.zeros((n*m * m*p, R))
            for r in range(R):
                outer = np.outer(U[:, r], V[:, r])
                A[:, r] = outer.flatten()
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                solved_W[k] = x
            except:
                break
        
        W = solved_W
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                if _check_decomposition(U, V, W, T):
                    return U, V, W
    
    return None

def _try_rank17_random_als(max_iters=30, attempts=20):
    """
    Try random initialization + ALS for rank 17, with half-integer rounding.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    start_time = time.time()
    deadline = start_time + 40  # 40 second budget
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        # Random integer initialization
        U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
        
        # Avoid zero columns
        for r in range(R):
            if np.all(U[:, r] == 0): U[0, r] = 1
            if np.all(V[:, r] == 0): V[0, r] = 1
            if np.all(W[:, r] == 0): W[0, r] = 1
        
        for it in range(max_iters):
            # ALS step for U
            U_new = np.zeros_like(U)
            for i in range(n*m):
                b = T[i].flatten()
                A = np.zeros((m*p * p*n, R))
                for r in range(R):
                    outer = np.outer(V[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i] = x
                except:
                    U_new[i] = U[i]
            U = U_new
            
            # ALS step for V
            V_new = np.zeros_like(V)
            for j in range(m*p):
                b = T[:, j, :].transpose(1, 0).flatten()
                A = np.zeros((n*m * p*n, R))
                for r in range(R):
                    outer = np.outer(U[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j] = x
                except:
                    V_new[j] = V[j]
            V = V_new
            
            # ALS step for W
            W_new = np.zeros_like(W)
            for k in range(p*n):
                b = T[:, :, k].flatten()
                A = np.zeros((n*m * m*p, R))
                for r in range(R):
                    outer = np.outer(U[:, r], V[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k] = x
                except:
                    W_new[k] = W[k]
            W = W_new
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-5:
                # Try to round to half-integers
                U_rounded = np.round(2 * U) / 2
                V_rounded = np.round(2 * V) / 2
                W_rounded = np.round(2 * W) / 2
                
                if _check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
                
                # Also check if already half-integer
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    if _check_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target()
    start_time = time.time()
    deadline = start_time + 120  # 120 second total budget
    
    # Try rank 17 with algebraic approach
    if time.time() < deadline - 30:
        result = _try_rank17_algebraic()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 17 with random ALS
    if time.time() < deadline - 20:
        result = _try_rank17_random_als()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 18
    U, V, W = _build_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
