"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1,2], k in [0,1,2].
    We need U (6 x r), V (9 x r), W (6 x r) such that:
      T[i,j,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c]
    
    Schoolbook gives rank 18. We attempt to find a lower rank decomposition.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        n, m, p = 2, 3, 3
        rank = len(U[0])
        # For each (i,j,k), check: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # T[i,j,k] should be 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def verify_decomp_exact(U, V, W):
        """Exact verification using Fraction arithmetic."""
        n, m, p = 2, 3, 3
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if val != 1:
                        return False
        return True
    
    # Generate schoolbook as baseline (rank 18)
    U_sb = [[0]*18 for _ in range(6)]
    V_sb = [[0]*18 for _ in range(9)]
    W_sb = [[0]*18 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j][col] = 1
                V_sb[j*3+k][col] = 1
                W_sb[k*2+i][col] = 1
                col += 1
    
    best_rank = 18
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find rank 16 decomposition
    # Use structured search with small integer coefficients
    
    import random
    random.seed(42)
    
    def try_rank(r, max_iters=3000, coeff_options=[-1, 0, 1]):
        """Try random search for given rank with small integer coefficients."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Generate random U, V, W with entries from coeff_options
            U = [[random.choice(coeff_options) for _ in range(r)] for _ in range(6)]
            V = [[random.choice(coeff_options) for _ in range(r)] for _ in range(9)]
            W = [[random.choice(coeff_options) for _ in range(r)] for _ in range(6)]
            
            if verify_decomp(U, V, W):
                # Double-check with exact arithmetic
                if verify_decomp_exact(U, V, W):
                    return (U, V, W)
        return None
    
    # Try rank 16
    result = try_rank(16, max_iters=2000)
    if result is not None:
        best_rank = 16
        best_decomp = result
    elif time.time() < deadline - 10:
        # Try rank 17
        result = try_rank(17, max_iters=2000)
        if result is not None:
            best_rank = 17
            best_decomp = result
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
