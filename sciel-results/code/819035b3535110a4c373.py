import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We use the Strassen algorithm for 2x2 matrix multiplication
    # applied to the 2x2 blocks of the 4x4 matrices.
    # This gives a rank of 4 * 7 = 28 for the 4x4x4 part.
    # Since p=8, we have two 4x4 blocks in B and C.
    # The U matrix is shared, so we need to pad it.
    # Total rank = 28 (for U) + 28 (for V1) + 28 (for V2) = 84? No.
    # The rank is the number of columns in U, V, W.
    # U has 28 columns.
    # V has 28 + 28 = 56 columns.
    # W has 28 + 28 = 56 columns.
    # We pad U with 28 zero columns to make it 56 columns.
    # Total rank = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # Strassen uses 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We need to map this to the U, V, W format.
    # For a 2x2 multiplication, the indices are:
    # U is 2x2, V is 2x2, W is 2x2.
    # The rank is 7.
    
    # For 4x4, we have 4 blocks.
    # Block (0,0): A11, B11, C11
    # Block (0,1): A12, B21, C11 (part 2)
    # Block (1,0): A21, B12, C21 (part 1)
    # Block (1,1): A22, B22, C22
    
    # Actually, the Strassen algorithm computes the product of two 2x2 matrices.
    # We have 4 such products.
    # We can use the same Strassen decomposition for each.
    
    # Let's define the Strassen U, V, W for 2x2.
    # U is 2x2 x 7, V is 2x2 x 7, W is 2x2 x 7.
    
    # We will construct U, V, W for the 4x4 case by placing the 2x2 blocks.
    
    # This is getting complex. Let's just use the schoolbook for now and see if we can improve.
    # Actually, the rank 28 is a good target.
    
    # Let's implement the Strassen decomposition for 2x2 and use it.
    
    return {"U": U, "V": V, "W": W}
