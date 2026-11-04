import time
import random
import fractions
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _construct_hybrid_strassen(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2 blocks and schoolbook for the rest.
    A: n x m (3x3)
    B: m x p (3x4)
    C: n x p (3x4)
    
    Blocks:
    A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    
    C11 (2x2), C12 (2x2), C21 (1x2), C22 (1x2)
    """
    if n != 3 or m != 3 or p != 4:
        return _schoolbook(n, m, p)
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen 2x2 * 2x2
    # M1 = (A11+A22)(B11+B22) -> (A00+A11)(B00+B11)
    # M2 = (A21+A22)B11 -> (A10+A11)B00 ... wait, indices are local.
    
    # Local indices for A11 (rows 0,1 cols 0,1)
    # a00=0, a01=1, a10=2, a11=3
    # Local indices for B11 (rows 0,1 cols 0,1)
    # b00=0, b01=1, b10=2, b11=3
    
    # Global indices for A
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # Global indices for B
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    
    # Global indices for C (p*n = 4*3 = 12)
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
    # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    # C[2,0]=8, C[2,1]=9, C[2,2]=10, C[2,3]=11
    
    # 1. A11 * B11 -> C11
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 3, 4
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 4, 5
    # C11: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 4, 5
    
    # Strassen Terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1) # A00 B00 C00
    add_term(0, 1, 1, 1, 1, 1) # A00 B01 C01
    add_term(4, 0, 4, 1, 1, 1) # A11 B00 C10
    add_term(4, 1, 5, 1, 1, 1) # A11 B01 C11
    # M2 = (A10+A11)B00
    add_term(3, 0, 4, 1, 1, 1) # A10 B00 C10
    add_term(4, 0, 5, 1, 1, 1) # A11 B00 C11
    # M3 = A00(B01-B11)
    add_term(0, 1, 1, 1, 1, 1) # A00 B01 C01
    add_term(0, 5, 5, 1, -1, 1) # A00 B11 C11
    # M4 = A11(B10-B00)
    add_term(4, 4, 4, 1, 1, 1) # A11 B10 C10
    add_term(4, 0, 5, 1, -1, 1) # A11 B00 C11
    # M5 = (A00+A01)B11
    add_term(0, 5, 4, 1, 1, 1) # A00 B11 C10
    add_term(1, 5, 5, 1, 1, 1) # A01 B11 C11
    # M6 = (A10-A00)(B00+B01)
    add_term(3, 0, 4, 1, 1, 1) # A10 B00 C10
    add_term(3, 1, 5, 1, 1, 1) # A10 B01 C11
    add_term(0, 0, 4, -1, 1, 1) # A00 B00 C10
    add_term(0, 1, 5, -1, 1, 1) # A00 B01 C11
    # M7 = (A01-A11)(B10+B11)
    add_term(1, 4, 4, 1, 1, 1) # A01 B10 C10
    add_term(1, 5, 5, 1, 1, 1) # A01 B11 C11
    add_term(4, 4, 4, -1, 1, 1) # A11 B10 C10
    add_term(4, 5, 5, -1, 1, 1) # A11 B11 C11
    
    # 2. A11 * B12 -> C12
    # A11: 0, 1, 3, 4
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 6, 7
    # C12: C[0,2], C[0,3], C[1,2], C[1,3] -> 2, 3, 6, 7
    
    # M1 = (A00+A11)(B02+B12)
    add_term(0, 2, 2, 1, 1, 1)
    add_term(0, 3, 3, 1, 1, 1)
    add_term(4, 6, 6, 1, 1, 1)
    add_term(4, 7, 7, 1, 1, 1)
    # M2 = (A10+A11)B02
    add_term(3, 2, 6, 1, 1, 1)
    add_term(4, 2, 7, 1, 1, 1)
    # M3 = A00(B03-B13)
    add_term(0, 3, 3, 1, 1, 1)
    add_term(0, 7, 7, 1, -1, 1)
    # M4 = A11(B12-B02)
    add_term(4, 6, 6, 1, 1, 1)
    add_term(4, 2, 7, 1, -1, 1)
    # M5 = (A00+A01)B12
    add_term(0, 6, 6, 1, 1, 1)
    add_term(1, 6, 7, 1, 1, 1)
    # M6 = (A10-A00)(B02+B03)
    add_term(3, 2, 6, 1, 1, 1)
    add_term(3, 3, 7, 1, 1, 1)
    add_term(0, 2, 6, -1, 1, 1)
    add_term(0, 3, 7, -1, 1, 1)
    # M7 = (A01-A11)(B12+B13)
    add_term(1, 6, 6, 1, 1, 1)
    add_term(1, 7, 7, 1, 1, 1)
    add_term(4, 6, 6, -1, 1, 1)
    add_term(4, 7, 7, -1, 1, 1)
    
    # 3. A12 * B21 -> C11
    # A12: A[0,2]=2, A[1,2]=5
    # B21: B[2,0]=8, B[2,1]=9
    # C11: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    add_term(2, 8, 0, 1, 1, 1)
    add_term(2, 9, 1, 1, 1, 1)
    add_term(5, 8, 4, 1, 1, 1)
    add_term(5, 9, 5, 1, 1, 1)
    
    # 4. A12 * B22 -> C12
    # A12: 2, 5
    # B22: B[2,2]=10, B[2,3]=11
    # C12: 2, 3, 6, 7
    add_term(2, 10, 2, 1, 1, 1)
    add_term(2, 11, 3, 1, 1, 1)
    add_term(5, 10, 6, 1, 1, 1)
    add_term(5, 11, 7, 1, 1, 1)
    
    # 5. A21 * B11 -> C21
    # A21: A[2,0]=6, A[2,1]=7
    # B11: 0, 1, 4, 5
    # C21: C[2,0]=8, C[2,1]=9
    add_term(6, 0, 8, 1, 1, 1)
    add_term(6, 1, 9, 1, 1, 1)
    add_term(7, 4, 8, 1, 1, 1)
    add_term(7, 5, 9, 1, 1, 1)
    
    # 6. A22 * B21 -> C21
    # A22: A[2,2]=8
    # B21: 8, 9
    # C21: 8, 9
    add_term(8, 8, 8, 1, 1, 1)
    add_term(8, 9, 9, 1, 1, 1)
    
    # 7. A21 * B12 -> C22
    # A21: 6, 7
    # B12: 2, 3, 6, 7
    # C22: C[2,2]=10, C[2,3]=11
    add_term(6, 2, 10, 1, 1, 1)
    add_term(6, 3, 11, 1, 1, 1)
    add_term(7, 6, 10, 1, 1, 1)
    add_term(7, 7, 11, 1, 1, 1)
    
    # 8. A22 * B22 -> C22
    # A22: 8
    # B22: 10, 11
    # C22: 10, 11
    add_term(8, 10, 10, 1, 1, 1)
    add_term(8, 11, 11, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try hybrid strassen
    result = _construct_hybrid_strassen(n, m, p)
    
    # Validate
    if _validate(result['U'], result['V'], result['W'], n, m, p, len(result['U'][0])):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
