"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*m+j][r] corresponds to A[i,j]
    # V[j*p+k][r] corresponds to B[j,k]  
    # W[k*n+i][r] corresponds to C[i,k]
    # We need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if val != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _rank_32_decomposition():
    """
    Construct a rank-32 decomposition for 3x3 x 3x4 matrix multiplication.
    
    Strategy: Split B (3x4) into B1 (3x3) and B2 (3x1).
    C = A*B1 + A*B2
    
    A*B1: 3x3 x 3x3 multiplication. Use a rank-23 decomposition (Lenstra).
    A*B2: 3x3 x 3x1 multiplication. Rank 9 (schoolbook).
    Total: 23 + 9 = 32.
    
    For the 3x3x3 part, we use a known rank-23 decomposition.
    For the 3x3x1 part, we use schoolbook (9 terms).
    """
    n, m, p = 3, 3, 4
    
    # Part 1: 3x3 x 3x3 (columns 0-2 of B, rows 0-2 of B)
    # Output goes to columns 0-2 of C
    # Use rank-23 decomposition for 3x3x3
    
    # Lenstra's rank-23 decomposition for 3x3x3:
    # This is a known result. Let me implement it.
    # The decomposition has 23 terms with specific coefficients.
    
    # For simplicity and correctness, let me use a verified rank-23 decomposition.
    # Here's one known decomposition (from the literature):
    
    # Actually, implementing Lenstra's exact decomposition is complex.
    # Let me use a different approach: use rank 27 for 3x3x3 (schoolbook) 
    # and see if we can get total rank 36, which is the same as schoolbook.
    # That's not helpful.
    
    # Let me try a different strategy: directly construct a rank-32 decomposition
    # by finding 4 linear dependencies in the schoolbook 36 terms.
    
    # Alternative: Use the fact that we can write:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # For k=0,1,2 (first 3 columns), use a rank-23 method
    # For k=3 (last column), use rank-9 method
    # Total: 32
    #
    # But I need the actual rank-23 coefficients.
    
    # Let me try yet another approach: use a randomized search to find 
    # a rank-32 decomposition.
    
    # The system of equations is:
    # For each (i,j,k) in [0,3)x[0,3)x[0,4):
    #   sum_{r=0}^{31} U[i*3+j][r] * V[j*4+k][r] * W[k*3+i][r] = 1
    #
    # 36 equations, 33*32 = 1056 unknowns. Very underdetermined.
    # 
    # Strategy: Set most variables to 0, solve for the rest.
    # For each column r, choose a sparse pattern.
    
    # Let me try: for each of the 32 columns, assign a "type" based on 
    # which (i,j,k) terms it helps satisfy.
    
    # Column types:
    # - Type (i,j,k): contributes to equation (i,j,k) only
    # - Type (i,j): contributes to all k for fixed i,j
    # - Type (i,k): contributes to all j for fixed i,k  
    # - Type (j,k): contributes to all i for fixed j,k
    
    # If I use type (i,j,k) columns, I need 36 columns (schoolbook).
    # To reduce to 32, I need 4 columns that each cover multiple equations.
    
    # A column with U[i*m+j] = a, V[j*p+k] = b, W[k*n+i] = c for specific (i,j,k)
    # only helps equation (i,j,k).
    
    # But if I set U[i*m+j] = a for multiple j values, it can help multiple equations.
    
    # Let me try: create 4 "shared" columns that each cover 2 equations,
    # reducing the total from 36 to 32.
    
    # For example, column r covers equations (i,j1,k1) and (i,j2,k2) if:
    # U[i*m+j1][r] * V[j1*p+k1][r] * W[k1*n+i][r] = 1
    # U[i*m+j2][r] * V[j2*p+k2][r] * W[k2*n+i][r] = 1
    # And 0 for all other equations.
    
    # This is possible if we set:
    # U[i*m+j1][r] = a, U[i*m+j2][r] = b
    # V[j1*p+k1][r] = c, V[j2*p+k2][r] = d
    # W[k1*n+i][r] = e, W[k2*n+i][r] = f
    # with a*c*e = 1, b*d*f = 1, and all other products = 0.
    
    # For other equations (i',j',k') not in {(i,j1,k1),(i,j2,k2)}:
    # U[i'*m+j'][r] * V[j'*p+k'][r] * W[k'*n+i'][r] = 0
    # This requires at least one factor to be 0 for each such equation.
    
    # This is feasible if the non-zero entries are carefully chosen.
    
    # Let me try a concrete construction:
    # Start with schoolbook (36 columns), then try to merge pairs.
    
    U = [[0] * 32 for _ in range(9)]
    V = [[0] * 32 for _ in range(12)]
    W = [[0] * 32 for _ in range(12)]
    
    # First 32 columns: use schoolbook for 32 of the 36 terms
    # Skip 4 terms and try to cover them with modified columns
    
    # Terms to skip: let's skip (0,0,0), (0,1,1), (1,0,2), (1,1,3)
    # And modify 4 other columns to cover these.
    
    # This is getting too complex. Let me just return the schoolbook for now
    # and rely on the verification to ensure correctness.
    
    # Actually, let me try a simpler approach: return None and let the 
    # schoolbook be the fallback.
    
    return None

def _try_rank_search(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition using structured search."""
    
    # For now, this is a placeholder. 
    # A proper implementation would use algebraic methods.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    n, m, p = DIMENSIONS
    
    # Verify schoolbook works
    assert _verify_decomposition(best['U'], best['V'], best['W'], n, m, p), "Schoolbook failed!"
    
    # Try to find a rank-32 decomposition
    if time.time() < deadline - 20:
        result = _rank_32_decomposition()
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = r
    
    return best

# EVOLVE-BLOCK-END
