import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 50 decomposition.
    # Split m into two parts of size 2.
    # For each part, we compute 2x2 * 2x7.
    # We split p=7 into 3 parts of size 2 and 1 part of size 1.
    # For each 2x2 * 2x2, use Strassen (rank 7).
    # For the 2x2 * 2x1, use schoolbook (rank 4).
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen matrices for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor indices.
    # A indices: i*2 + j. i in 0,1. j in 0,1.
    # B indices: j*7 + k. j in 0,1. k in 0,1 (for the 2x2 block).
    # W indices: k*2 + i. k in 0,1. i in 0,1.
    
    def add_strassen(a_off, b_off, w_off, start_col):
        # a_off: offset for A indices (i*2+j)
        # b_off: offset for B indices (j*7+k)
        # w_off: offset for W indices (k*2+i)
        
        # M1
        U[a_off + 0][start_col] = 1; V[b_off + 0][start_col] = 1; W[w_off + 0][start_col] = 1
        U[a_off + 1][start_col] = 1; V[b_off + 1][start_col] = 1; W[w_off + 0][start_col] += 1
        U[a_off + 2][start_col] = 1; V[b_off + 0][start_col] += 1; W[w_off + 2][start_col] = 1
        U[a_off + 3][start_col] = 1; V[b_off + 1][start_col] += 1; W[w_off + 2][start_col] += 1
        # This is getting complicated to write manually.
        # Better to use a helper or just define the 7 terms explicitly.
        
        return start_col + 7

    # This manual construction is error-prone.
    # Let's just use the schoolbook rank 56 for safety if I can't verify 50 quickly.
    # But 50 is known.
    
    # Actually, I can just return the schoolbook rank 56.
    # Score 1.0.
    # But I want to improve.
    
    # Let's try to return rank 50.
    # I will write a generic function to add a 2x2*2x2 Strassen block.
    
    pass
