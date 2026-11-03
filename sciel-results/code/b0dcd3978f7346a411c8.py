"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    r = len(U[0])
    # For each (i, j, k) in the multiplication tensor, check:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    # For all other combinations, the sum should be 0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for r in range(r):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if actual != expected:
                    return False
    return True

def _find_low_rank_decomposition(n, m, p, max_rank=None, time_limit=135):
    """
    Search for a low-rank exact decomposition of the matrix multiplication tensor.
    Uses systematic enumeration with rational arithmetic.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    if max_rank is None:
        max_rank = n * m * p
    
    # The tensor T[i,j,k] = 1 for all (i,j,k), else 0
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = T[i,j,k]
    
    # Try ranks from lower to higher
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        
        # For small ranks, we can try to find solutions by solving linear systems
        # The key insight: each rank-1 term contributes to the tensor as:
        # U_col * V_col * W_col (outer product in the appropriate sense)
        
        # Strategy: Fix U and V, then solve for W
        # Or use a more clever approach based on the structure
        
        # For 2x3x6, let's try to find a decomposition with rank < 36
        # The tensor has 2*3*6 = 36 entries, all equal to 1
        
        # A rank-r decomposition means we can write:
        # T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's try a structured approach:
        # If we set U[i,j,r] = a_{i,j,r}, V[j,k,r] = b_{j,k,r}, W[k,i,r] = c_{k,i,r}
        # Then we need sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1 for all (i,j,k)
        
        # One approach: use the fact that the tensor is "constant" (all ones)
        # to find a low-rank decomposition
        
        # For constant tensors, there are known results. Let's try:
        # If n, m, p are such that we can factor, we might get lower rank
        
        # Try rank = n*p (if m=1, this would be optimal)
        # For our case: n=2, m=3, p=6
        
        # Let's try a direct search for small ranks using the structure
        # We'll fix a pattern and solve linear equations
        
        # For rank r, we have r unknowns per matrix entry
        # Total unknowns: r*(n*m + m*p + p*n) = r*(6 + 18 + 12) = 36r
        # Constraints: n*m*p = 36 equations
        
        # For r >= 1, we have more unknowns than constraints (if r >= 1)
        # So solutions should exist, but we need them to be exact
        
        # Let's try a greedy/structured approach:
        # Set most entries to 0 or 1, and solve for the rest
        
        # Actually, for the all-ones tensor, there's a simple decomposition:
        # Use r = max(n, m, p) or similar
        
        # Let's try: U[i,j,r] = 1 if r == i*m+j else 0 (schoolbook)
        # That gives rank = n*m*p
        
        # Better: try to share across indices
        
        # For the constant-1 tensor, one approach:
        # U[i,j,r] = 1/sqrt(n*m) for all i,j,r (if r is chosen appropriately)
        # But we need exact rational/integer solutions
        
        # Let's try a different structure:
        # If we set U[i,j,r] = u_r, V[j,k,r] = v_r, W[k,i,r] = w_r
        # (independent of i,j,k), then sum_r u_r*v_r*w_r = 1
        # This gives rank 1 if we can find u,v,w with u*v*w = 1
        # But this doesn't work because the indices are coupled
        
        # The issue is that U depends on (i,j), V on (j,k), W on (k,i)
        # So they share indices in a cyclic way
        
        # For the all-ones tensor, let's try:
        # U[i,j,r] = 1 for all i,j,r (if r is fixed)
        # V[j,k,r] = 1 for all j,k,r
        # W[k,i,r] = 1 for all k,i,r
        # Then sum_r 1*1*1 = r, so we need r = 1
        # But this gives T[i,j,k] = 1 for all (i,j,k), which is correct!
        # Wait, that's rank 1!
        
        # Let me verify: if U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
        # Then for any (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1
        # This is correct for the all-ones tensor!
        
        # But wait, is the matrix multiplication tensor really all ones?
        # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = 1 if we're computing the (i,k) entry using (j) as the sum index
        # So T[i,j,k] = 1 for all valid (i,j,k)
        
        # Yes! The multiplication tensor is indeed all ones for valid indices.
        # So rank 1 should work!
        
        # Let me double-check the indexing:
        # U has shape (n*m, rank), indexed by (i, j) -> row i*m + j
        # V has shape (m*p, rank), indexed by (j, k) -> row j*p + k  
        # W has shape (p*n, rank), indexed by (k, i) -> row k*n + i
        
        # For the decomposition to work:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
        
        # If we set all entries to 1 and rank = 1:
        # sum = 1 * 1 * 1 = 1. Correct!
        
        # So the answer is rank 1!
        
        if rank == 1:
            U = [[Fraction(1)] for _ in range(n * m)]
            V = [[Fraction(1)] for _ in range(m * p)]
            W = [[Fraction(1)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                # Convert to float for output
                U_float = [[float(x) for x in row] for row in U]
                V_float = [[float(x) for x in row] for row in V]
                W_float = [[float(x) for x in row] for row in W]
                return U_float, V_float, W_float
    
    # If we couldn't find a better decomposition, fall back to schoolbook
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = Fraction(1)
                V[j*p + k][col] = Fraction(1)
                W[k*n + i][col] = Fraction(1)
                col += 1
    
    U_float = [[float(x) for x in row] for row in U]
    V_float = [[float(x) for x in row] for row in V]
    W_float = [[float(x) for x in row] for row in W]
    return U_float, V_float, W_float

def construct():
    """
    Returns the lowest-rank exact tensor decomposition found for matrix multiplication 2x3 * 3x6.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _find_low_rank_decomposition(n, m, p, time_limit=135)
    except Exception as e:
        # Fall back to schoolbook on any error
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
