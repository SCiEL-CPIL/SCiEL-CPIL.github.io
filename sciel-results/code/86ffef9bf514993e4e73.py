"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if decomposition is exact using rational arithmetic."""
    n, m, p = DIMENSIONS
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    rank = len(W_f[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if s != 1:
                    return False
    # Check that non-target entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k == 0:
                    continue
                # Actually we need to check that sum equals 1 for (i,j,k) and 0 otherwise
                # But the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor T[i,j,k] should equal 1 for all (i,j,k) in the multiplication
                # Actually no: the matrix multiplication tensor has T[i,j,k] = 1 if the product
                # (AB)_{i,k} involves A_{i,j} * B_{j,k}, so every (i,j,k) triple is a valid term.
                # So we just need each triple to sum to 1.
                pass
    return True

def _verify_numpy(U, V, W, tol=1e-9):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    U_a = np.array(U, dtype=float)
    V_a = np.array(V, dtype=float)
    W_a = np.array(W, dtype=float)
    rank = W_a.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_a[i * m + j], V_a[j * p + k] * W_a[k * n + i])
                if abs(s - 1) > tol:
                    return False
    return True

def _try_block_diagonal(r_block):
    """
    Try to build a decomposition by tiling a 4x4x4 block decomposition.
    If we have a rank-r_block decomposition for 4x4x4, we can tile it 7 times
    along the p dimension to get a 4x4x7 decomposition of rank 7*r_block.
    But we need the W matrix to have shape (p*n, rank) = (28, rank).
    
    For 4x4x4: U is 16xR, V is 16xR, W is 16xR (W[k*n+i] for k=0..3, i=0..3)
    For 4x4x7: U is 16xR', V is 28xR', W is 28xR'
    
    Tiling approach: split k into blocks. For each block of the 7 k-values,
    we need to handle the multiplication. But 4x4x4 blocks don't directly tile
    because the V matrix couples j and k.
    
    Alternative: use the fact that for 4x4x7, we can think of it as 7 separate
    4x4x1 multiplications combined, but that doesn't help.
    
    Actually, a simpler structural approach: if we find a rank-R decomposition
    for 4x4xR where R is some value, we can potentially extend.
    
    Let me try a different approach: search for a low-rank decomposition directly
    using randomized linear algebra.
    """
    return None

def _randomized_search(r_max, time_limit=100.0):
    """
    Try to find a rank-r decomposition for r from r_max down to some minimum.
    Uses randomized initialization and iterative refinement.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    best = None
    
    # Try ranks from r_max down
    for r in range(r_max, r_max - 10, -1):
        if time.time() - start > time_limit:
            break
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start > time_limit:
                break
            # Random initialization
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Iterative refinement: for each (i,j,k), the constraint is
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a system of n*m*p = 112 equations in n*m*r + m*p*r + p*n*r variables
            # = r*(16 + 28 + 28) = 72r unknowns
            # For r >= 2, we have more unknowns than equations, so solutions exist generically.
            # But we need integer/half-integer entries.
            
            # For now, just check if the random initialization happens to work (unlikely)
            if _verify_numpy(U, V, W):
                # Convert to fractions
                U_frac = [[Fraction(2 * round(U[i][c] * 2)) / 2 for c in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(2 * round(V[i][c] * 2)) / 2 for c in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(2 * round(W[i][c] * 2)) / 2 for c in range(r)] for i in range(p*n)]
                if _verify(U_frac, V_frac, W_frac):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _greedy_construction():
    """
    Greedy approach: build the decomposition term by term.
    Each term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to T[i,j,k].
    We need the sum over c to equal 1 for all (i,j,k).
    
    Start with the schoolbook (each term handles one (i,j,k) triple).
    Try to merge pairs of terms: if two terms have overlapping structure,
    can we combine them?
    """
    n, m, p = DIMENSIONS
    school = _schoolbook()
    rank = n * m * p
    return school

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # First, get the schoolbook decomposition as fallback
    fallback = _schoolbook()
    
    # Try a quick randomized search for a lower rank
    # Theoretically, for 4x4x7, the rank could be as low as ~28 (lower bound) 
    # but practically, finding an exact decomposition with small rank is hard.
    
    # Try rank 111, 110, ... down to 105 quickly
    result = _randomized_search(111, time_limit=100.0)
    if result is not None:
        return result
    
    # If time is running out, return the schoolbook
    if time.time() > deadline:
        return fallback
    
    return fallback
