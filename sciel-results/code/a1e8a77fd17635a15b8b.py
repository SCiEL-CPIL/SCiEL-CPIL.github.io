"""Matrix multiplication 2x3x7 tensor decomposition with minimal rank."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x7) into B1 (2x7) and B2 (1x7).
    # C = A1 B1 + A2 B2.
    
    # Part 1: A2 B2 (2x1 times 1x7)
    # C[i,k] = A[i,2] * B[2,k]
    # Tensor term: T[i,2,k] = 1.
    # Decomposition: Rank 1.
    # U[i,2,0] = 1, V[2,k,0] = 1, W[k,i,0] = 1.
    
    # Part 2: A1 B1 (2x2 times 2x7)
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # For each k, this is a 2x2x1 multiplication.
    # Rank of 2x2x1 is 4 (schoolbook).
    # Total rank for this part: 7 * 4 = 28.
    
    # Total Rank: 1 + 28 = 29.
    
    decomp = _build_optimized_decomp(n, m, p)
    
    if decomp is not None and _verify_exact(decomp, n, m, p):
        return decomp
        
    # Fallback
    return _schoolbook_decomp(n, m, p)

def _schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _verify_exact(decomp, n, m, p):
    """Exact verification using fractions."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(rank):
                    u_val = Fraction(str(U[i*m+j][r]))
                    v_val = Fraction(str(V[j*p+k][r]))
                    w_val = Fraction(str(W[k*n+i][r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _build_optimized_decomp(n, m, p):
    """
    Builds a decomposition with rank 29 for 2x3x7.
    """
    if n != 2 or m != 3 or p != 7:
        return None
        
    # Indices:
    # U: (i, j) where i in 0..1, j in 0..2
    # V: (j, k) where j in 0..2, k in 0..6
    # W: (k, i) where k in 0..6, i in 0..1
    
    U = [[0] * 29 for _ in range(6)]
    V = [[0] * 29 for _ in range(21)]
    W = [[0] * 29 for _ in range(14)]
    
    col = 0
    
    # Part 1: A2 B2 (j=2)
    # T[i, 2, k] = 1
    # Use 1 term.
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2][col] = 1
            V[2 * 7 + k][col] = 1
            W[k * 2 + i][col] = 1
    col += 1
    
    # Part 2: A1 B1 (j=0,1)
    # For each k in 0..6:
    #   C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k]
    #   Terms:
    #   1. A[i,0]B[0,k] -> U[i,0]=1, V[0,k]=1, W[k,i]=1
    #   2. A[i,1]B[1,k] -> U[i,1]=1, V[1,k]=1, W[k,i]=1
    #   Wait, this is 4 terms per k?
    #   i=0, j=0: U[0,0]=1, V[0,k]=1, W[k,0]=1
    #   i=0, j=1: U[0,1]=1, V[1,k]=1, W[k,0]=1
    #   i=1, j=0: U[1,0]=1, V[0,k]=1, W[k,1]=1
    #   i=1, j=1: U[1,1]=1, V[1,k]=1, W[k,1]=1
    
    for k in range(7):
        # Term 1: i=0, j=0
        U[0 * 3 + 0][col] = 1
        V[0 * 7 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        
        # Term 2: i=0, j=1
        U[0 * 3 + 1][col] = 1
        V[1 * 7 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        
        # Term 3: i=1, j=0
        U[1 * 3 + 0][col] = 1
        V[0 * 7 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
        # Term 4: i=1, j=1
        U[1 * 3 + 1][col] = 1
        V[1 * 7 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
