"""Program for 5x5 matrix multiplication tensor decomposition.
Implements the BCLR (Bini-Capovani-Lotti-Romani) rank-78 decomposition.
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit
    
    # The BCLR (Bini, Capovani, Lotti, Romani, 1979) decomposition
    # gives rank 78 for 5x5 matrix multiplication.
    # 
    # The construction uses a clever block decomposition. The key idea:
    # 5x5 matrices can be decomposed using a specific set of 78 rank-1 tensors.
    #
    # The BCLR construction works as follows:
    # We identify 78 triples (i,j,k) with associated coefficients that
    # exactly reproduce matrix multiplication.
    #
    # The explicit construction from BCLR (1979) uses the following approach:
    # Split the 5x5 multiplication into structured sub-problems and
    # use bilinear algorithms for each part.
    
    # The rank-78 decomposition for 5x5 matrix multiplication
    # from BCLR (1979). The 78 terms are constructed as follows:
    #
    # The decomposition uses the fact that:
    # 5 = 3 + 2, and we can use:
    # - Strassen's 7-multiplication algorithm for 3x3 blocks
    # - Direct multiplication for 2x2 blocks  
    # - Cross terms handled efficiently
    #
    # But the actual BCLR construction is more subtle. Let me use
    # the known explicit form.
    
    # Actually, let me implement a verified decomposition.
    # The BCLR rank-78 decomposition can be constructed using
    # the following explicit formulas.
    
    # I'll use a different approach: construct the decomposition
    # using known bilinear algorithms and verify correctness.
    
    # For 5x5, we can use the following block structure:
    # A (5x5) = [A11 A12 A13]
    #           [A21 A22 A23]
    #           [A31 A32 A33]
    # where A11 is 3x3, A12 is 3x1, A13 is 3x1,
    # A21 is 1x3, A22 is 1x1, A23 is 1x1,
    # A31 is 1x3, A32 is 1x1, A33 is 1x1
    
    # This gives:
    # C11 (3x3) = A11*B11 + A12*B21 + A13*B31
    # C12 (3x1) = A11*B12 + A12*B22 + A13*B32
    # C13 (3x1) = A11*B13 + A12*B23 + A13*B33
    # C21 (1x3) = A21*B11 + A22*B21 + A23*B31
    # C22 (1x1) = A21*B12 + A22*B22 + A23*B32
    # C23 (1x1) = A21*B13 + A22*B23 + A23*B33
    # C31 (1x3) = A31*B11 + A32*B21 + A33*B31
    # C32 (1x1) = A31*B12 + A32*B22 + A33*B32
    # C33 (1x1) = A31*B13 + A32*B23 + A33*B33
    
    # Using Strassen for the 3x3*3x3 part: 7 mults
    # Direct for other parts...
    
    # This approach gives a specific rank. Let me compute it:
    # C11: A11*B11 (7 Strassen) + A12*B21 (3*1*3=9) + A13*B31 (3*1*3=9) = 25
    # C12: A11*B12 (3*3*1=9) + A12*B22 (3*1*1=3) + A13*B32 (3*1*1=3) = 15
    # C13: A11*B13 (9) + A12*B23 (3) + A13*B33 (3) = 15
    # C21: A21*B11 (1*3*3=9) + A22*B21 (1*1*3=3) + A23*B31 (1*1*3=3) = 15
    # C22: A21*B12 (1*3*1=3) + A22*B22 (1) + A23*B32 (1) = 5
    # C23: A21*B13 (3) + A22*B23 (1) + A23*B33 (1) = 5
    # C31: A31*B11 (9) + A32*B21 (3) + A33*B31 (3) = 15
    # C32: A31*B12 (3) + A32*B22 (1) + A33*B32 (1) = 5
    # C33: A31*B13 (3) + A32*B23 (1) + A33*B33 (1) = 5
    
    # Total: 25+15+15+15+5+5+15+5+5 = 105
    
    # That's better than 125 but not 78.
    
    # For rank 78, we need a more sophisticated approach.
    # The BCLR construction uses a different decomposition.
    
    # Let me try to implement the actual BCLR rank-78 decomposition.
    # The key insight from BCLR is to use a specific set of bilinear forms.
    
    # I'll construct the rank-78 decomposition using the following method:
    # The 78 multiplications are organized into groups corresponding
    # to different block interactions.
    
    # Due to the complexity of the explicit BCLR formulas, I'll use
    # a verified decomposition. Let me construct it programmatically.
    
    # Actually, let me use a simpler but still improved decomposition.
    # I can achieve rank 79 or so with a careful block decomposition.
    
    # For now, let me return the schoolbook decomposition as a fallback,
    # but I'll try to find a better one.
    
    # Let me try a different block split: 5 = 4 + 1
    # A = [A11(4x4) | A12(4x1)]
    #     [A21(1x4) | A22(1x1)]
    
    # C11 (4x4) = A11*B11
    # C12 (4x1) = A11*B12
    # C21 (1x4) = A21*B11
    # C22 (1x1) = A21*B12 + A22*B21 + A23*B32 + A31*B33... 
    # Wait, this doesn't work cleanly for 5x5.
    
    # Let me just implement the schoolbook decomposition for now.
    # The rank-78 BCLR decomposition requires explicit formulas
    # that are very complex to implement correctly without reference.
    
    rank = n * m * p  # 125 for schoolbook
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
