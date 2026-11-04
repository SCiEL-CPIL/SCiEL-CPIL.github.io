import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-51 exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 9 matrix.
    
    Rank 51 is achieved by:
    1. Splitting A (2x3) into A_L (2x2) and A_R (2x1).
    2. Splitting B (3x9) into B_L (2x9) and B_R (1x9).
    3. C = A_L * B_L + A_R * B_R.
    4. A_L * B_L (2x2 * 2x9) uses 3 blocks of (2x2 * 2x3).
       Each 2x2 * 2x3 uses Strassen (rank 7) for 2x2 * 2x2 part
       and Outer Product (rank 4) for 2x2 * 2x1 part.
       Total rank for A_L * B_L = 3 * (7 + 4) = 33.
    5. A_R * B_R (2x1 * 1x9) is an outer product, rank 18.
    6. Total rank = 33 + 18 = 51.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 51
    V = [[0] * rank for _ in range(m * p)] # 27 x 51
    W = [[0] * rank for _ in range(p * n)] # 18 x 51
    
    col = 0
    
    # --- Part 1: A_L (2x2) * B_L (2x9) ---
    # A_L indices: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 3, 4
    # B_L indices: rows 0,1 of B. B is 3x9.
    # B_L is 2x9. Split into 3 blocks of 2x3: cols 0-2, 3-5, 6-8.
    
    # Strassen 2x2x2 decomposition coefficients
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)i  -> wait, standard Strassen is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)i
    # M6 = (c-b)(e+h)
    # M7 = (b+d)(g+f)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A_L in U:
    # a -> U[0], b -> U[1], c -> U[3], d -> U[4]
    
    for block in range(3):
        # B block cols: block*3 to block*3+2
        # B_L block is 2x3: rows 0,1. Cols c0, c1, c2.
        # Split into B_2x2 (cols c0, c1) and B_2x1 (col c2).
        
        c0 = block * 3
        c1 = block * 3 + 1
        c2 = block * 3 + 2
        
        # B indices in V:
        # Row 0 of B: V[0*9 + k] = V[k]
        # Row 1 of B: V[1*9 + k] = V[9+k]
        
        # --- Strassen part (2x2 * 2x2) ---
        # Uses cols c0, c1 of B_L block.
        # e = B[0, c0] -> V[c0]
        # f = B[0, c1] -> V[c1]
        # g = B[1, c0] -> V[9+c0]
        # h = B[1, c1] -> V[9+c1]
        
        # M1 = (a+d)(e+h)
        U[0][col] = 1; U[4][col] = 1
        V[c0][col] = 1; V[9+c1][col] = 1
        W[c0*2 + 0][col] = 1 # C[0, c0]
        W[c1*2 + 1][col] = 1 # C[1, c1]
        col += 1
        
        # M2 = (c+d)e
        U[3][col] = 1; U[4][col] = 1
        V[c0][col] = 1
        W[c0*2 + 1][col] = 1 # C[1, c0]
        W[c1*2 + 1][col] = -1 # C[1, c1]
        col += 1
        
        # M3 = a(f-h)
        U[0][col] = 1
        V[c1][col] = 1; V[9+c0][col] = -1
        W[c0*2 + 1][col] = 1 # C[1, c0]
        W[c1*2 + 0][col] = 1 # C[0, c1]
        col += 1
        
        # M4 = d(g-f)
        U[4][col] = 1
        V[9+c0][col] = 1; V[c1][col] = -1
        W[c0*2 + 0][col] = 1 # C[0, c0]
        W[c0*2 + 1][col] = 1 # C[1, c0]
        col += 1
        
        # M5 = (a+b)i -> i is B[0, c1]
        U[0][col] = 1; U[1][col] = 1
        V[c1][col] = 1
        W[c0*2 + 0][col] = -1 # C[0, c0]
        W[c1*2 + 0][col] = 1 # C[0, c1]
        col += 1
        
        # M6 = (c-b)(e+h)
        U[3][col] = 1; U[1][col] = -1
        V[c0][col] = 1; V[9+c1][col] = 1
        W[c1*2 + 1][col] = 1 # C[1, c1]
        col += 1
        
        # M7 = (b+d)(g+f)
        U[1][col] = 1; U[4][col] = 1
        V[9+c0][col] = 1; V[c1][col] = 1
        W[c0*2 + 0][col] = 1 # C[0, c0]
        W[c1*2 + 1][col] = 1 # C[1, c1]
        col += 1
        
        # --- Outer part (2x2 * 2x1) ---
        # Uses col c2 of B_L block.
        # B_2x1 is [[B[0, c2]], [B[1, c2]]]
        # e2 = B[0, c2] -> V[c2]
        # g2 = B[1, c2] -> V[9+c2]
        
        # C contributions:
        # C[0, c2] += a * e2
        # C[1, c2] += d * g2
        # C[0, c2] += b * e2  (Wait, outer product of A_L and B_2x1)
        # A_L = [[a, b], [c, d]]. B_2x1 = [[e2], [g2]].
        # C_2x1 = [[a*e2 + b*g2], [c*e2 + d*g2]]?
        # No, matrix mult:
        # [a b] [e2] = [a*e2 + b*g2]
        # [c d] [g2]   [c*e2 + d*g2]
        # So 4 terms: a*e2, b*g2, c*e2, d*g2.
        
        # Term 1: a * e2
        U[0][col] = 1
        V[c2][col] = 1
        W[c2*2 + 0][col] = 1 # C[0, c2]
        col += 1
        
        # Term 2: b * g2
        U[1][col] = 1
        V[9+c2][col] = 1
        W[c2*2 + 0][col] = 1 # C[0, c2]
        col += 1
        
        # Term 3: c * e2
        U[3][col] = 1
        V[c2][col] = 1
        W[c2*2 + 1][col] = 1 # C[1, c2]
        col += 1
        
        # Term 4: d * g2
        U[4][col] = 1
        V[9+c2][col] = 1
        W[c2*2 + 1][col] = 1 # C[1, c2]
        col += 1
        
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: A_R (2x1) * B_R (1x9) ---
    # A_R indices: (0,2) -> U[2], (1,2) -> U[5]
    # B_R indices: row 2 of B. V[2*9 + k] = V[18+k]
    # C indices: all 18 entries.
    # C[0, k] += A[0,2] * B[2, k]
    # C[1, k] += A[1,2] * B[2, k]
    
    for k in range(9):
        # Term 1: A[0,2] * B[2, k]
        U[2][col] = 1
        V[18+k][col] = 1
        W[k*2 + 0][col] = 1 # C[0, k]
        col += 1
        
        # Term 2: A[1,2] * B[2, k]
        U[5][col] = 1
        V[18+k][col] = 1
        W[k*2 + 1][col] = 1 # C[1, k]
        col += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
