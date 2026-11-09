"""Matrix multiplication 2x3 * 3x6: rank-minimal tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: sum_t U[a*m+b,t] * V[c*p+d,t] * W[k*n+i,t] == delta(a,i)*delta(c,b)*delta(d,k)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                            expected = (1.0 if (a==i and c==b and d==k) else 0.0)
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with W[k*n+i, t] indexing.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook rank = 36
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
# EVOLVE-BLOCK-END
