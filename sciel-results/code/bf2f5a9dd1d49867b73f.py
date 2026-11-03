"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # safety margin
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find rank r such that T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and the decomposition satisfies:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known lower bounds: For 3x3 * 3x5, the rank is at least max(n*m, m*p, p*n) = max(9,15,15) = 15
    # But actually the trivial bound is n*m*p = 45. 
    # The best known for 3x3x3 is 23 (Strassen-like), but for 3x3x5 we need to search.
    # 
    # Let's try a random search / optimization approach.
    # We'll use a simple iterative method: start with random factors and try to reduce rank.
    # Actually, let's try to find a low-rank decomposition using numerical optimization.
    
    # Strategy: Try ranks from a lower bound upward.
    # Lower bound: The rank must be at least the maximum of the dimensions of the slices.
    # More precisely, rank >= max(n*m, m*p, p*n) is not quite right.
    # The true lower bound is related to the slice rank. For T[i,j,k]=1, the slice rank is max(n*m, m*p, p*n).
    # Wait, slice rank is different from tensor rank. But tensor rank >= slice rank.
    # Slice rank of the all-ones tensor is max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
    # So rank >= 15.
    
    # Let's try to find a decomposition of rank 15, 16, 17, ... up to some limit.
    # We'll use a random initialization and then do a few iterations of gradient descent
    # or a simple alternating minimization.
    
    # Actually, for such a small problem, we can try a more systematic approach.
    # Let's use the fact that for the all-ones tensor, there might be structured decompositions.
    
    # One known approach: For multiplying A (n x m) by B (m x p), 
    # the rank is at most n*m*p, but can be lower.
    # For 3x3 * 3x5, let's try random search with verification.
    
    def verify(U, V, W, r):
        """Verify that the decomposition is exact."""
        # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # and check it equals 1 for all i,j,k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W, r):
        """Verify with exact arithmetic (fractions)."""
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0, 1)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != Fraction(1, 1):
                        return False
        return True
    
    # Try ranks from 15 upward
    # Use random search with small integer entries
    
    for r in range(15, 46):
        if time.time() > deadline:
            break
        # Try random integer matrices with entries in {-1, 0, 1}
        for attempt in range(500):
            if time.time() > deadline:
                break
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.randint(-1, 1) for _ in range(r)] for _ in range(p*n)]
            
            # Quick check: for each (i,j,k), the sum should be 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U, "V": V, "W": W}
    
    # If we didn't find a low-rank solution, fall back to schoolbook
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
# EVOLVE-BLOCK-END
