import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    # ... (same as before) ...
    pass

def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    # ... (same as before) ...
    pass

def _build_coefficient_matrix(U, V, W, n=2, m=4, p=8):
    # ... (same as before) ...
    pass

def _find_null_vector_numpy(U, V, W, n=2, m=4, p=8):
    """Find a non-trivial null space vector using NumPy."""
    R = len(U[0])
    A = _build_coefficient_matrix(U, V, W, n, m, p)
    
    # Use SVD to find null space
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    # The null space is the last (R - rank) rows of Vt
    # Find the rank
    tol = 1e-10
    rank = np.sum(s > tol)
    
    if rank >= R:
        return None
        
    # Get the null space vectors
    # Vt is (R, R). The rows corresponding to zero singular values span the null space.
    # Actually, SVD of A (M x R) gives Vt (R x R).
    # The last R-rank rows of Vt are the null space basis.
    null_basis = Vt[rank:, :]
    
    # Pick a vector with large integer-like components
    # Try to find a vector in the null space that has integer or half-integer entries
    # We can iterate over combinations or just pick the one with largest norm
    # For now, pick the first one
    v = null_basis[0]
    
    # Normalize to have largest component 1 or -1
    max_idx = np.argmax(np.abs(v))
    if v[max_idx] == 0:
        return None
        
    v = v / v[max_idx]
    
    # Round to nearest integer or half-integer
    v_rounded = np.round(v * 2) / 2.0
    
    # Check if it's close to the rounded version
    if not np.allclose(v, v_rounded, atol=1e-6):
        # Try other vectors in the basis
        found = False
        for i in range(1, len(null_basis)):
            v = null_basis[i]
            max_idx = np.argmax(np.abs(v))
            if v[max_idx] != 0:
                v = v / v[max_idx]
                v_rounded = np.round(v * 2) / 2.0
                if np.allclose(v, v_rounded, atol=1e-6):
                    found = True
                    break
        if not found:
            # Try linear combinations? Too complex.
            # Just return the best candidate and let exact verification fail
            pass
            
    return v_rounded

def _try_reduce_rank(U, V, W, n=2, m=4, p=8, deadline=None):
    if deadline is not None and time.time() > deadline:
        return None
        
    R = len(U[0])
    
    # Quick check with numpy
    A = _build_coefficient_matrix(U, V, W, n, m, p)
    rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    
    if rank_A >= R:
        return None
        
    # Find null vector using numpy
    c_float = _find_null_vector_numpy(U, V, W, n, m, p)
    if c_float is None:
        return None
        
    # Convert to Fractions for exact verification
    c = [Fraction(int(round(x * 2))) / 2 for x in c_float]
    
    # Verify exact null space
    valid = True
    for eq in range(len(A)):
        dot = 0
        for r in range(R):
            dot += A[eq, r] * c[r]
        if dot != 0:
            valid = False
            break
            
    if not valid:
        return None
        
    # Construct new decomposition
    # ... (same as before) ...
    pass

def construct():
    # ... (same as before) ...
    pass
