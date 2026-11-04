"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction as F

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # We seek a rank-minimal exact tensor decomposition for 2x3 * 3x4.
    # The known lower bound is 17 (from 14+3? or 17+? No, the lower bound for 2x3x4 is 17? 
    # Actually, the rank of 2x3x4 is known to be 18 or 19?
    # Wait, let's look at the lower bound: max(2*3 + 3*4 - 1, 3*4 + 4*2 - 1, 4*2 + 2*3 - 1) = max(11, 15, 11) = 15?
    # No, the simple lower bound is not that strong.
    # The rank of 2x3x4 is 18? Or 19?
    # Actually, it is known that the rank of 2x3x4 is 18?
    # Let's try to construct a rank 18 or 19 decomposition.
    
    # Strategy: Block decomposition
    # A is 2x3, B is 3x4.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x4) and B2 (1x4).
    # C = A1 B1 + A2 B2.
    
    # For A1 B1 (2x2 * 2x4), we can use Strassen-like ideas or just direct.
    # Direct is 2*2*4 = 16 terms.
    # Strassen for 2x2 * 2x2 is 7 terms.
    # So A1 B1a (2x2 * 2x2) uses 7 terms.
    # A1 B1b (2x2 * 2x2) uses 7 terms.
    # Total for A1 B1 is 14 terms.
    # A2 B2 (2x1 * 1x4) uses 2*1*4 = 8 terms.
    # Total rank = 14 + 8 = 22.
    
    # Can we do better?
    # Maybe split A into 1x3 + 1x3? No.
    # Maybe use a different partition?
    # What if we split B into 3x2 + 3x2?
    # A (2x3) * B1 (3x2) uses 2*3*2 = 12 terms (schoolbook) or maybe less?
    # Rank of 2x3x2 is 6? No, 2*3*2 = 12?
    # Wait, the rank of 2x3x2 is 6?
    # Let's check: 2x3x2 tensor.
    # It is equivalent to 2x2x3.
    # Rank of 2x2x3 is 5? Or 6?
    # Actually, the rank of 2x2x3 is 5? No, it's 5?
    # Wait, the rank of 2x2x2 is 7.
    # The rank of 2x2x3 is 5?
    # No, the rank of 2x2x3 is 5?
    # Let's assume we can find a rank 18 decomposition.
    
    # Actually, the known rank of 2x3x4 is 18?
    # Let's try to build a rank 18 decomposition.
    # If we can't find it, we return the rank 22 one, which is better than 24.
    
    # Let's implement the rank 22 decomposition first.
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A[i, j] with i in 0..1, j in 0..2.
    # B[j, k] with j in 0..2, k in 0..3.
    # C[i, k] with i in 0..1, k in 0..3.
    
    # Split A: A1 is rows 0..1, cols 0..1 (2x2). A2 is rows 0..1, col 2 (2x1).
    # Split B: B1 is rows 0..1, cols 0..3 (2x4). B2 is row 2, cols 0..3 (1x4).
    
    # A1 B1:
    # A1 is 2x2, B1 is 2x4.
    # Split B1 into B1a (2x2, cols 0..1) and B1b (2x2, cols 2..3).
    # A1 B1a: 2x2 * 2x2 -> 7 Strassen terms.
    # A1 B1b: 2x2 * 2x2 -> 7 Strassen terms.
    
    # A2 B2:
    # A2 is 2x1, B2 is 1x4.
    # 2x1 * 1x4 -> 8 terms (schoolbook).
    
    # Total 7 + 7 + 8 = 22.
    
    # Let's code the Strassen part for A1 B1a.
    # A1 = [[a00, a01], [a10, a11]]
    # B1a = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11) b00
    # M3 = a00 (b11-b01)
    # M4 = a11 (b00-b10)
    # M5 = (a00+a01) b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b10+b11)
    
    # C1a = [[M1+M4-M5-M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to map these to U, V, W columns.
    # Each column c corresponds to a product term.
    # U[i*m+j, c] is the coefficient of A[i,j] in the term.
    # V[j*p+k, c] is the coefficient of B[j,k] in the term.
    # W[k*n+i, c] is the coefficient of the term in C[i,k].
    
    # Note: W index is k*n + i.
    
    # Term 0: M1
    c = 0
    # A part: a00 + a11 -> U[0*m+0][c]=1, U[1*m+1][c]=1
    U[0*3+0][c] = 1
    U[1*3+1][c] = 1
    # B part: b00 + b11 -> V[0*4+0][c]=1, V[1*4+1][c]=1
    V[0*4+0][c] = 1
    V[1*4+1][c] = 1
    # W part: contributes to C1a[0,0], C1a[1,1]
    # C1a[0,0] is C[0,0] -> W[0*2+0][c] = 1
    # C1a[1,1] is C[1,1] -> W[1*2+1][c] = 1
    W[0*2+0][c] = 1
    W[1*2+1][c] = 1
    
    # Term 1: M2
    c = 1
    U[1*3+0][c] = 1
    U[1*3+1][c] = 1
    V[0*4+0][c] = 1
    # C1a[1,0] is C[1,0] -> W[0*2+1][c] = 1
    W[0*2+1][c] = 1
    
    # Term 2: M3
    c = 2
    U[0*3+0][c] = 1
    V[1*4+1][c] = 1
    V[0*4+1][c] = -1
    # C1a[0,1] is C[0,1] -> W[1*2+0][c] = 1
    W[1*2+0][c] = 1
    
    # Term 3: M4
    c = 3
    U[1*3+1][c] = 1
    V[0*4+0][c] = 1
    V[1*4+0][c] = -1
    # C1a[0,0] is C[0,0] -> W[0*2+0][c] = 1
    # C1a[1,0] is C[1,0] -> W[0*2+1][c] = 1
    W[0*2+0][c] = 1
    W[0*2+1][c] = 1
    
    # Term 4: M5
    c = 4
    U[0*3+0][c] = 1
    U[0*3+1][c] = 1
    V[1*4+1][c] = 1
    # C1a[0,0] is C[0,0] -> W[0*2+0][c] = -1
    # C1a[0,1] is C[0,1] -> W[1*2+0][c] = 1
    W[0*2+0][c] = -1
    W[1*2+0][c] = 1
    
    # Term 5: M6
    c = 5
    U[1*3+0][c] = 1
    U[0*3+0][c] = -1
    V[0*4+0][c] = 1
    V[1*4+0][c] = 1
    # C1a[1,1] is C[1,1] -> W[1*2+1][c] = 1
    W[1*2+1][c] = 1
    
    # Term 6: M7
    c = 6
    U[0*3+1][c] = 1
    U[1*3+1][c] = -1
    V[1*4+0][c] = 1
    V[1*4+1][c] = 1
    # C1a[0,0] is C[0,0] -> W[0*2+0][c] = -1
    W[0*2+0][c] = -1
    
    # Now A1 B1b (cols 2..3 of B)
    # B1b = [[b02, b03], [b12, b13]]
    # Same structure, but B indices are k=2,3.
    
    # Term 7: M1'
    c = 7
    U[0*3+0][c] = 1
    U[1*3+1][c] = 1
    V[0*4+2][c] = 1
    V[1*4+3][c] = 1
    # C1b[0,0] is C[0,2] -> W[2*2+0][c] = 1
    # C1b[1,1] is C[1,3] -> W[3*2+1][c] = 1
    W[2*2+0][c] = 1
    W[3*2+1][c] = 1
    
    # Term 8: M2'
    c = 8
    U[1*3+0][c] = 1
    U[1*3+1][c] = 1
    V[0*4+2][c] = 1
    # C1b[1,0] is C[1,2] -> W[2*2+1][c] = 1
    W[2*2+1][c] = 1
    
    # Term 9: M3'
    c = 9
    U[0*3+0][c] = 1
    V[1*4+3][c] = 1
    V[0*4+3][c] = -1
    # C1b[0,1] is C[0,3] -> W[3*2+0][c] = 1
    W[3*2+0][c] = 1
    
    # Term 10: M4'
    c = 10
    U[1*3+1][c] = 1
    V[0*4+2][c] = 1
    V[1*4+2][c] = -1
    # C1b[0,0] is C[0,2] -> W[2*2+0][c] = 1
    # C1b[1,0] is C[1,2] -> W[2*2+1][c] = 1
    W[2*2+0][c] = 1
    W[2*2+1][c] = 1
    
    # Term 11: M5'
    c = 11
    U[0*3+0][c] = 1
    U[0*3+1][c] = 1
    V[1*4+3][c] = 1
    # C1b[0,0] is C[0,2] -> W[2*2+0][c] = -1
    # C1b[0,1] is C[0,3] -> W[3*2+0][c] = 1
    W[2*2+0][c] = -1
    W[3*2+0][c] = 1
    
    # Term 12: M6'
    c = 12
    U[1*3+0][c] = 1
    U[0*3+0][c] = -1
    V[0*4+2][c] = 1
    V[1*4+2][c] = 1
    # C1b[1,1] is C[1,3] -> W[3*2+1][c] = 1
    W[3*2+1][c] = 1
    
    # Term 13: M7'
    c = 13
    U[0*3+1][c] = 1
    U[1*3+1][c] = -1
    V[1*4+2][c] = 1
    V[1*4+3][c] = 1
    # C1b[0,0] is C[0,2] -> W[2*2+0][c] = -1
    W[2*2+0][c] = -1
    
    # Now A2 B2 (A col 2, B row 2)
    # A2 = [[a02], [a12]]
    # B2 = [b20, b21, b22, b23]
    # C2[i, k] = a_i2 * b_2k
    
    for i in range(2):
        for k in range(4):
            c = 14 + i*4 + k
            U[i*3+2][c] = 1
            V[2*4+k][c] = 1
            W[k*2+i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
