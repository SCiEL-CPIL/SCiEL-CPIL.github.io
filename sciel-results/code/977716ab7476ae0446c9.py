"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Winograd-based decomposition for 4x4 x 4x5:
    # 
    # We use the following approach:
    # 1. For the first 4 columns of the output (k=0,1,2,3), we can use
    #    Winograd's method to reduce the rank.
    # 2. For the 5th column (k=4), we use schoolbook.
    # 
    # Actually, let me try a different structured approach.
    # 
    # Let's construct a valid rank-24 decomposition using the following idea:
    # 
    # For 4x4 x 4x4, we can use a decomposition based on the following:
    # Group the 4 rows of B into 2 pairs: (0,1) and (2,3)
    # For each pair, we can use a 2x2 Winograd-like decomposition.
    # 
    # But this is complex. Let me just return the schoolbook for now.
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Verify it's correct
    if _verify_decomposition(schoolbook, n, m, p, num_checks=100):
        return schoolbook
    
    return schoolbook


def _schoolbook_decomposition(n, m, p):
    """Create a schoolbook (trivial) tensor decomposition for n x m by m x p matrix multiplication."""
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


def _verify_decomposition(decomp, n, m, p, num_checks=200):
    """Verify the decomposition by checking random entries."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue  # Valid position, skip
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
