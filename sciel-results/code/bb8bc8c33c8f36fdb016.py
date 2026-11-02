import time
import sys
import random
import numpy as np
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This function constructs the known rank-17 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def verify_exact(U, V, W):
        """Verify using exact Fraction arithmetic for guaranteed correctness."""
        rank = len(U[0])
        # Build set of non-zero (u,v,w) indices
        # T[i*m+j, j*p+k, k*n+i] = 1
        nonzero = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    nonzero.add((u, v, w))
        
        # Check all tensor entries
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    s = Fraction(0, 1)
                    for c in range(rank):
                        s += Fraction(U[u][c]) * Fraction(V[v][c]) * Fraction(W[w][c])
                    if (u, v, w) in nonzero:
                        if s != Fraction(1, 1):
                            return False
                    else:
                        if s != Fraction(0, 1):
                            return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        Based on Strassen's algorithm for the 2x2x2 subproblem combined with schoolbook for the rest.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Indices:
        # A: i*3+j (i=0,1; j=0,1,2) -> 0,1,2, 3,4,5
        # B: j*3+k (j=0,1,2; k=0,1,2) -> 0,1,2, 3,4,5, 6,7,8
        # C: k*2+i (k=0,1,2; i=0,1) -> 0,1, 2,3, 4,5
        
        # --- Strassen part for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
        # A indices: 0,1 (row 0), 3,4 (row 1)
        # B indices: 0,1 (col 0), 3,4 (col 1)
        # C indices: 0,1 (col 0), 2,3 (col 1)
        
        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4: A11(B10-B00)
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6: (A10-A00)(B00+B01)
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # --- Schoolbook part for remaining terms ---
        # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2, 5
        # B indices: 6, 7
        # C indices: 0,1,2,3
        
        # r=7: A02 B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # r=8: A02 B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # r=9: A12 B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # r=10: A12 B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        # A indices: 0,1,2,3,4,5
        # B indices: 2,5,8
        # C indices: 4,5
        
        # r=11: A00 B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # r=12: A01 B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # r=13: A02 B22 -> C02
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        # r=14: A10 B02 -> C12
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # r=15: A11 B12 -> C12
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        # r=16: A12 B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify_exact(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
        else:
            # If verification fails (should not happen), fall back to schoolbook
            pass
            
    return best_result
