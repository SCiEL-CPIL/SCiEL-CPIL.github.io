import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 28 (for 4x4x4) + 28 (for 4x4x4) + 16 (for 4x4x1) = 72
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Define Strassen terms for 2x2 blocks
    # Blocks are (i1,i2) x (j1,j2)
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # Block 1: k in 0..3
    # Strassen steps
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will iterate through the 7 steps and 4 inner products
    # For each step, we define the coefficients for A and B
    # Then for each of the 4 schoolbook terms in the 2x2 multiplication,
    # we expand the products of sums.
    
    # This is getting complex to hardcode.
    # Let's use a more direct approach:
    # Generate the 28 terms for 4x4x4 by explicitly listing the indices.
    
    # Actually, let's just use the rank-144 schoolbook if the complexity is too high?
    # No, 72 is much better.
    
    # Let's implement the 28 terms for the first block (k=0..3)
    # and the second block (k=4..7)
    # and the 16 terms for the third block (k=8)
    
    # To save time, I will write a helper that takes a list of (a_idx, b_idx, c_idx) tuples
    # and adds them.
    
    # But I need to generate the tuples.
    
    # Let's define the 2x2 block indices
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B00 (k=0,1): j in 0,1; k in 0,1
    # B01 (k=2,3): j in 0,1; k in 2,3
    # B10 (k=0,1): j in 2,3; k in 0,1
    # B11 (k=2,3): j in 2,3; k in 2,3
    
    # C00 (k=0,1): k in 0,1; i in 0,1
    # C01 (k=2,3): k in 2,3; i in 0,1
    # C10 (k=0,1): k in 0,1; i in 2,3
    # C11 (k=2,3): k in 2,3; i in 2,3
    
    # Strassen Formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to expand these into scalar multiplications.
    # Each M is a 2x2 matrix product.
    # M = X * Y
    # X, Y are 2x2 matrices.
    # M[i,j] = X[i,0]Y[0,j] + X[i,1]Y[1,j]
    
    # Let's define a function to expand a product of two "block sums"
    
    pass
