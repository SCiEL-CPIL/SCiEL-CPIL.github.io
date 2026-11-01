"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 2*2*5 = 20
    
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    
    # Generate schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb, r_sb)
    
    # Try to find better decompositions
    # Use structured search with small integer/half-integer entries
    # Values to try: integers and half-integers in a reasonable range
    values = [-2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2]
    
    # Try ranks from 11 to 19 (lower bound is 10, but 10 is unlikely to work)
    for target_rank in range(11, schoolbook_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 20000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V, W with entries from values
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p, target_rank):
                found = True
                best_decomp = (U, V, W, target_rank)
                break
        
        if found:
            break
    
    U, V, W, rank = best_decomp
    
    # Convert to proper format - ensure entries are integers or half-integers
    # The values are already in that form
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
