"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the matrix multiplication tensor which is all 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_integer_or_half_integer(val, tol=1e-9):
    """Check if a value is an integer or half-integer."""
    # Multiply by 2 and check if it's close to an integer
    doubled = val * 2
    nearest = round(doubled)
    return abs(doubled - nearest) < tol

def _try_solve_for_W(U, V, n, m, p, R):
    """
    Given fixed U and V (each R columns), solve for W.
    The system is: for all i,j,k:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is linear in W. Let's set up the linear system.
    Unknowns: W[0..p*n-1, 0..R-1], total p*n*R unknowns.
    Equations: n*m*p = 120 equations.
    
    For equation (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Note that W[k*n+i, r] appears only in equations where the (i,k) pair matches.
    So for a fixed (i,k), the equations over j are:
        sum_j [sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]] ... 
    Wait, no. Each (i,j,k) is a separate equation.
    
    Let me reorganize. The unknowns are indexed by (ik, r) where ik = k*n+i.
    For equation (i,j,k), the coefficient of W[ik, r] is U[i*m+j, r] * V[j*p+k, r].
    
    So we have a system A * x = b where:
    - x is a vector of length p*n*R (flattened W)
    - A has n*m*p rows and p*n*R columns
    - b is all 1s
    
    This is an underdetermined system (120 equations, 24R unknowns).
    We need to find a solution where all entries are integer or half-integer.
    
    Let's use numpy to find the minimum-norm solution or a particular solution.
    """
    num_eqs = n * m * p
    num_unk = p * n * R
    
    # Build the coefficient matrix
    # A[row, col] where row = (i,j,k) flattened, col = (ik, r) flattened
    # ik = k*n + i, so col = ik * R + r
    # row = (i * m + j) * p + k
    
    A = np.zeros((num_eqs, num_unk), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = (i * m + j) * p + k
                ik = k * n + i
                for r in range(R):
                    col = ik * R + r
                    A[row, col] = U[i * m + j][r] * V[j * p + k][r]
    
    b = np.ones(num_eqs, dtype=float)
    
    # Solve the least-squares problem
    try:
        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution satisfies the equations
    if residuals is not None and len(residuals) > 0:
        if residuals[0] > 1e-6:
            return None
    
    # Reshape x to W
    W = [[0.0] * R for _ in range(p * n)]
    for ik in range(p * n):
        for r in range(R):
            W[ik][r] = x[ik * R + r]
    
    # Check if all entries are integer or half-integer
    for ik in range(p * n):
        for r in range(R):
            if not _is_integer_or_half_integer(W[ik][r]):
                return None
    
    # Convert to exact integer/half-integer representation
    for ik in range(p * n):
        for r in range(R):
            W[ik][r] = round(W[ik][r] * 2) / 2.0
    
    return W

def _generate_sparse_UV(n, m, p, R, rng):
    """
    Generate random sparse U and V matrices with R columns.
    Each column of U has a few non-zero entries (integer or half-integer).
    Each column of V has a few non-zero entries (integer or half-integer).
    """
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    
    for r in range(R):
        # Choose 2-4 random positions for U column r
        num_nonzero_U = rng.randint(2, min(4, n * m))
        positions_U = rng.sample(range(n * m), num_nonzero_U)
        for pos in positions_U:
            # Value is 1 or -1 or 0.5 or -0.5
            U[pos][r] = rng.choice([1, -1, 0.5, -0.5])
        
        # Choose 2-4 random positions for V column r
        num_nonzero_V = rng.randint(2, min(4, m * p))
        positions_V = rng.sample(range(m * p), num_nonzero_V)
        for pos in positions_V:
            V[pos][r] = rng.choice([1, -1, 0.5, -0.5])
    
    return U, V

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x5 @ 5x6.
    Strategy:
    1. Try to find a decomposition with rank < 120 by fixing sparse U, V patterns
       and solving for W via linear algebra.
    2. Fall back to schoolbook if no better rank is found.
    """
    start_time = time.time()
    deadline = 110.0  # Keep a safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better rank
    # Start from a reasonable upper bound and go down
    # Theoretical minimum is max(n*m, m*p, p*n) = 30
    # Try ranks from 100 down to 40
    for R in range(100, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Try several random sparse patterns
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(R * 1000 + attempt)
            U, V = _generate_sparse_UV(n, m, p, R, rng)
            
            W = _try_solve_for_W(U, V, n, m, p, R)
            if W is not None:
                # Verify the decomposition
                if _verify_decomposition(U, V, W, n, m, p):
                    result = {"U": U, "V": V, "W": W}
                    return result
    
    return best_result

# EVOLVE-BLOCK-END
