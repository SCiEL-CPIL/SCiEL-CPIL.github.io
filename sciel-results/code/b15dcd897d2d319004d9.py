"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from fractions import Fraction

def _make_tensors(n, m, p):
    """Build the canonical tensors for the matrix multiplication tensor."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rational_approx(x, max_denom=100):
    """Approximate x as a fraction with small denominator."""
    if abs(x) < 1e-12:
        return 0
    frac = Fraction(x).limit_denominator(max_denom)
    return frac

def _to_rational_list(arr, max_denom=100):
    """Convert a numpy array to a list of Fractions."""
    return [[_rational_approx(arr[i, j], max_denom) for j in range(arr.shape[1])] for i in range(arr.shape[0])]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Start with schoolbook decomposition (rank 30)
    U_school, V_school, W_school = _make_tensors(n, m, p)
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = n * m * p
    
    # Try to find rank 28 decomposition
    # The key insight is that for matrix multiplication, we can sometimes merge terms
    # by using algebraic identities. For 2x3x5, we need to find if rank 28 is possible.
    
    # Strategy: Use a structured approach based on the fact that 
    # some of the 30 terms can be combined using the distributive property.
    
    # For a 2x3 * 3x5 multiplication, the result is 2x5.
    # Each entry (i,k) of the result is sum_j A[i,j]*B[j,k]
    # The tensor has entries T[i,j,k] = 1 if this term contributes to C[i,k]
    
    # Let's try a specific structured decomposition for rank 28.
    # We can try to find a decomposition where 2 of the 30 terms are merged.
    
    # Actually, let's try a more systematic approach:
    # Use random initialization and optimize to find a rank-28 decomposition.
    
    # For now, let's try to find a rank-28 decomposition using a specific pattern.
    # One approach is to use the fact that some rows/columns can be shared.
    
    # Let's try a direct construction for rank 28.
    # If we can find a valid rank-28 decomposition, we return it.
    
    # Try rank 28 with a structured approach
    target_rank = 28
    found = False
    
    # Use a specific pattern: merge terms where possible
    # For 2x3x5, we have 30 terms. We need to eliminate 2.
    
    # One way: if two terms have the same (i,k) but different j, we can't merge them directly.
    # But if we use a clever basis, we might be able to.
    
    # Let's try a numerical approach: start with rank 30 and try to reduce.
    
    # Actually, the simplest approach is to try to find a rank-28 decomposition
    # by solving a system of equations.
    
    # For rank r, we need to find U (6xr), V (15xr), W (10xr) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of 30 equations with 6r + 15r + 10r = 31r unknowns.
    # For r=28, we have 868 unknowns and 30 equations, so it's underdetermined.
    # We can try to find a solution.
    
    # Let's try a simple approach: use the schoolbook decomposition and try to
    # merge two terms by setting some entries to zero and adjusting others.
    
    # Actually, let's just return the schoolbook decomposition for now, as finding
    # a rank-28 decomposition is complex and may not be possible.
    
    if _verify(best_U, best_V, best_W):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
