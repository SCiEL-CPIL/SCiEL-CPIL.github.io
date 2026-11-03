"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Verify exact decomposition. Returns True if valid."""
    r = len(U[0])
    # Check positive conditions: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for cc in range(r):
                    s += U[a][cc] * V[b][cc] * W[c][cc]
                if s != 1:
                    return False
    # Check negative conditions: for (a,b,c) not forming valid triple, sum = 0
    # Sample some
    for _ in range(200):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        i_c, k_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        s = Fraction(0)
        for cc in range(r):
            s += U[a][cc] * V[b][cc] * W[c][cc]
        if s != 0:
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by 1 using a simple perturbation approach
    # This is unlikely to work but let's try
    
    # For now, just return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
