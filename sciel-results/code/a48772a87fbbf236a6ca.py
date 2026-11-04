"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys
import json

DIMENSIONS = [2, 3, 3]

def _fraction_to_number(frac):
    """Convert Fraction to int or float for JSON serialization."""
    if frac.denominator == 1:
        return int(frac.numerator)
    return float(frac)

def _convert_matrix_to_json_safe(matrix):
    """Convert a matrix of Fractions to a matrix of ints/floats."""
    return [[_fraction_to_number(x) for x in row] for row in matrix]

def _snap_to_half_integer(matrix, tol=1e-6):
    """
    Snap matrix entries to the nearest multiple of 0.5 if within tolerance.
    Returns a list of lists of Fractions if successful, else None.
    """
    result = []
    for row in matrix:
        r = []
        for val in row:
            # Check if close to k/2
            # k = round(2 * val)
            k = round(2 * val)
            approx_val = k / 2.0
            if abs(val - approx_val) < tol:
                r.append(Fraction(k, 2))
            else:
                return None
        result.append(r)
    return result

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _random_small_rational_decomposition(rank, n, m, p, rng):
    """
    Generate random U, V with entries in {-1, -0.5, 0, 0.5, 1} and solve for W.
    Check if W can be snapped to half-integers.
    """
    # Define allowed values
    allowed_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    U = np.array([[rng.choice(allowed_vals) for _ in range(rank)] for _ in range(n*m)])
    V = np.array([[rng.choice(allowed_vals) for _ in range(rank)] for _ in range(m*p)])
    
    # Avoid trivial zero rows/cols which make the system singular or ill-conditioned
    # Ensure at least one non-zero in each row/col? 
    # For now, just solve.
    
    num_eqs = n*m*p
    num_unk = p*n*rank
    
    if num_unk < num_eqs:
        return None
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = (i * m + j) * p + k
                for c in range(rank):
                    unk_idx = (k * n + i) * rank + c
                    A[eq_idx, unk_idx] = U[i*m+j, c] * V[j*p+k, c]
    
    try:
        w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
        
    W = w_flat.reshape((p*n, rank))
    
    # Try to snap W to half-integers
    W_r = _snap_to_half_integer(W, tol=1e-5)
    if W_r is None:
        return None
        
    # Also snap U and V to half-integers (they are already, but convert to Fraction)
    U_r = _snap_to_half_integer(U, tol=1e-9)
    V_r = _snap_to_half_integer(V, tol=1e-9)
    
    if U_r is None or V_r is None:
        return None

    if _verify_decomposition_exact(U_r, V_r, W_r, n, m, p):
        # Convert to JSON-safe types immediately to avoid issues later
        return _convert_matrix_to_json_safe(U_r), _convert_matrix_to_json_safe(V_r), _convert_matrix_to_json_safe(W_r)
    
    return None

def _try_rank(rank, n, m, p, max_attempts=200, time_limit=None):
    """Try to find a rank-r decomposition."""
    rng = np.random.default_rng(42 + rank)
    
    for attempt in range(max_attempts):
        if time_limit is not None:
            if time.time() > time_limit:
                break
        
        result = _random_small_rational_decomposition(rank, n, m, p, rng)
        if result is not None:
            U, V, W = result
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high
    # Theoretical lower bound for 2x3x3 is 11? 
    # Let's try from 11 upwards. 
    # Actually, let's try a few lower ones just in case, but 11 is the known minimum for this shape?
    # Strassen 2x2x2 is 7. 
    # 2x3x3 can be viewed as 2x2x3 + 2x1x3? 
    # Let's just scan 10 to 18.
    
    for rank in range(10, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(rank, n, m, p, max_attempts=500, time_limit=deadline - 15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
    
    # Final safety check: ensure all values are finite numbers
    try:
        json.dumps({"U": U, "V": V, "W": W})
    except (TypeError, ValueError):
        # If serialization fails, force conversion to floats
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        
    return {"U": U, "V": V, "W": W}
