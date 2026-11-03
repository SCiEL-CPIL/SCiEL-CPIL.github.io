"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Build decomposition using Strassen for 2x2x2 blocks.
    # Split the 4x4 middle as 2x2 blocks, and 7 as 2+2+3.
    # 
    # A (2x4) = [A0 A1] where A0, A1 are 2x2
    # B (4x7) = [B00 B01 B02; B10 B11 B12] where B00,B01,B10,B11 are 2x2, B02,B12 are 2x3
    #
    # C = A0*B00 + A1*B10  (for cols 0-1)
    #   + A0*B01 + A1*B11  (for cols 2-3)  
    #   + A0*B02 + A1*B12  (for cols 4-6)
    #
    # Each 2x2*2x2 product uses Strassen (rank 7)
    # Each 2x2*2x3 product: split 2x3 as 2x2 + 2x1, use Strassen + schoolbook = 7+4 = 11
    #
    # Total rank: 7 + 7 + 11 = 25
    #
    # But we need to verify this is a valid tensor decomposition.
    # The key insight: the multiplication tensor for the full problem is the
    # direct sum of the tensors for each block, so ranks add.
    
    # Let me construct this explicitly.
    
    # Strassen decomposition for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # C00 = M1 - M3 + M5 + M7
    # C01 = M4 + M5
    # C10 = M1 + M2
    # C11 = M2 - M4 + M5 + M6
    # where:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = ae
    # M6 = (a+b)h
    # M7 = (c-d)(e+f)
    
    # For each Strassen term, U, V, W entries are ±1 or 0.
    
    # Let me define the Strassen factor matrices for a 2x2x2 block.
    # U is 4x7 (for A entries: a,b,c,d mapped to indices 0,1,2,3)
    # V is 4x7 (for B entries: e,f,g,h mapped to indices 0,1,2,3)
    # W is 4x7 (for C entries: C00,C01,C10,C11 mapped to indices 0,1,2,3)
    
    # M1 = (a+d)(e+h): U=[1,0,1,0], V=[1,0,0,1], W=[1,0,0,0]
    # M2 = (c+d)h: U=[0,0,1,1], V=[0,0,0,1], W=[0,0,1,0]
    # M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[-1,0,0,0]
    # M4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0], W=[0,-1,0,0]
    # M5 = ae: U=[1,0,0,0], V=[1,0,0,0], W=[0,0,0,1]
    # M6 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[0,0,0,1]
    # M7 = (c-d)(e+f): U=[0,0,1,-1], V=[1,1,0,0], W=[1,0,0,0]
    
    # Wait, let me double-check C00 = M1 - M3 + M5 + M7
    # M1 contributes 1 to C00, M3 contributes -1, M5 contributes 0, M7 contributes 1.
    # So C00 = M1 - M3 + 0 + M7. That matches.
    
    # C01 = M4 + M5: M4 contributes -1? No, M4 = d(g-e), and C01 should be bf+dh.
    # Hmm, I think I have the Strassen formulas slightly off. Let me use a standard reference.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = ae
    # M6 = (a+b)h
    # M7 = (c-d)(e+f)
    #
    # C00 = M1 - M3 + M5 - M7  (let me verify: (a+d)(e+h) - a(f-h) + ae - (c-d)(e+f))
    # = ae+ah+de+dh - af+ah+ae - ce-cf+de+df
    # = 2ae+2ah+2de+dh-af-ce-cf+df... this doesn't simplify to ae+bg.
    
    # I think I'm confusing myself. Let me just use a known correct Strassen decomposition.
    
    # Actually, for this task, let me just implement a verified decomposition.
    # I'll use the schoolbook as a fallback and try to construct a better one.
    
    # For now, return schoolbook to ensure validity.
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
# EVOLVE-BLOCK-END
