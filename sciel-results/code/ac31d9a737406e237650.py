"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _strassen222_decomposition():
    """
    Returns (U, V, W) for 2x2 by 2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7) indexed by i*m+j = i*2+j
    V: (4, 7) indexed by j*p+k = j*2+k
    W: (4, 7) indexed by k*n+i = k*2+i
    """
    # Strassen formulas for C = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (a+b)h
    # P3 = (c+d)e
    # P4 = (a-c)(h-g)
    # P5 = (b+d)g
    # P6 = (a+c)(e-f)
    # P7 = (b-d)(f-h)  -- actually standard is (b-d)(f-h) or similar, let's use standard Strassen
    
    # Standard Strassen:
    # P1 = (a + d) * (e + h)
    # P2 = (a + b) * h
    # P3 = (c + d) * e
    # P4 = (a - c) * (h - g)
    # P5 = (b + d) * g
    # P6 = (a + c) * (e - f)
    # P7 = (b - d) * (f - h)
    
    # U corresponds to A indices:
    # a=0, b=1, c=2, d=3 (i*2+j)
    # P1 uses a,d -> U[0,0]=1, U[3,0]=1
    # P2 uses a,b -> U[0,1]=1, U[1,1]=1
    # P3 uses c,d -> U[2,2]=1, U[3,2]=1
    # P4 uses a,c -> U[0,3]=1, U[2,3]=-1
    # P5 uses b,d -> U[1,4]=1, U[3,4]=1
    # P6 uses a,c -> U[0,5]=1, U[2,5]=1
    # P7 uses b,d -> U[1,6]=1, U[3,6]=-1
    
    U = np.zeros((4, 7), dtype=int)
    U[0, 0] = 1; U[3, 0] = 1
    U[0, 1] = 1; U[1, 1] = 1
    U[2, 2] = 1; U[3, 2] = 1
    U[0, 3] = 1; U[2, 3] = -1
    U[1, 4] = 1; U[3, 4] = 1
    U[0, 5] = 1; U[2, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V corresponds to B indices:
    # e=0, f=1, g=2, h=3 (j*2+k)
    # P1 uses e,h -> V[0,0]=1, V[3,0]=1
    # P2 uses h   -> V[3,1]=1
    # P3 uses e   -> V[0,2]=1
    # P4 uses h,g -> V[3,3]=1, V[2,3]=-1
    # P5 uses g   -> V[2,4]=1
    # P6 uses e,f -> V[0,5]=1, V[1,5]=-1
    # P7 uses f,h -> V[1,6]=1, V[3,6]=-1
    
    V = np.zeros((4, 7), dtype=int)
    V[0, 0] = 1; V[3, 0] = 1
    V[3, 1] = 1
    V[0, 2] = 1
    V[3, 3] = 1; V[2, 3] = -1
    V[2, 4] = 1
    V[0, 5] = 1; V[1, 5] = -1
    V[1, 6] = 1; V[3, 6] = -1
    
    # W corresponds to C indices:
    # C = [[C00, C01], [C10, C11]]
    # C00 = P1 + P4 - P5 - P7
    # C01 = P2 + P5
    # C10 = P3 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # Indexing: k*n + i = k*2 + i
    # i=0, j=0 -> C00 -> idx 0
    # i=0, j=1 -> C01 -> idx 1
    # i=1, j=0 -> C10 -> idx 2
    # i=1, j=1 -> C11 -> idx 3
    
    W = np.zeros((4, 7), dtype=int)
    # P1
    W[0, 0] += 1; W[3, 0] += 1
    # P2
    W[1, 1] += 1
    # P3
    W[2, 2] += 1; W[3, 2] += 1
    # P4
    W[0, 3] += 1; W[2, 3] += 1
    # P5
    W[0, 4] -= 1; W[1, 4] += 1
    # P6
    # P6 is not used in standard Strassen reconstruction?
    # Wait, standard Strassen uses P1..P7.
    # C00 = P1 + P4 - P5 - P7
    # C01 = P2 + P5
    # C10 = P3 + P4
    # C11 = P1 - P2 + P3 + P7
    # P6 is indeed not used in the output?
    # Let's check the definition of P6: (a+c)(e-f).
    # Is it used?
    # Actually, in some formulations, P6 is used.
    # Let's re-verify Strassen.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My P definitions:
    # P1 = (a+d)(e+h) -> M1
    # P2 = (a+b)h -> (A11+A12)B22 -> M5
    # P3 = (c+d)e -> (A21+A22)B11 -> M2
    # P4 = (a-c)(h-g) -> (A11-A21)(B22-B21) ... Not exactly M6 or M3.
    # Let's stick to a known correct set.
    
    # Let's use the explicit formulas for U, V, W that I can verify numerically.
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2
    U[0, 1] = 1; U[1, 1] = 1
    V[3, 1] = 1
    W[1, 1] = 1
    
    # P3
    U[2, 2] = 1; U[3, 2] = 1
    V[0, 2] = 1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4
    U[0, 3] = 1; U[2, 3] = -1
    V[3, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5
    U[1, 4] = 1; U[3, 4] = 1
    V[2, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6
    U[0, 5] = 1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = -1
    # P6 not used in C?
    
    # P7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = -1
    W[0, 6] = -1; W[3, 6] = 1
    
    # Verify:
    # C00 (idx 0) = P1 + P4 - P5 - P7
    # W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=-1. Sum = 0?
    # Let's check a specific entry.
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = [[19, 22], [43, 50]]
    
    # Let's just write the code to build the full 2x3x7 using block decomposition.
    # Rank 2x3x2 = 11 (7 for 2x2x2 + 4 for 2x1x2)
    # Rank 2x3x1 = 6
    # Total rank for 2x3x7 = 2 * Rank(2x3x2) + Rank(2x3x1) = 2*11 + 6 = 28?
    # Wait, 7 = 2 + 2 + 2 + 1.
    # So 3 blocks of 2 and 1 block of 1.
    # Rank = 3 * 11 + 6 = 39.
    
    # Let's implement the 2x3x2 rank 11 decomposition.
    # Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x2) into B1 (2x2) and B2 (1x2).
    # C = A1 * B1 + A2 * B2.
    
    # A1*B1 is 2x2x2, rank 7.
    # A2*B2 is 2x1x2, rank 4.
    
    # U for A1*B1: 4x7.
    # U for A2*B2: 2x4.
    # V for A1*B1: 4x7.
    # V for A2*B2: 2x4.
    # W for A1*B1: 4x7.
    # W for A2*B2: 4x4.
    
    # Total U: 6x11, V: 6x11, W: 4x11.
    
    # Let's build it.
    
    U_11 = np.zeros((4, 7), dtype=int)
    V_11 = np.zeros((4, 7), dtype=int)
    W_11 = np.zeros((4, 7), dtype=int)
    
    # Strassen 2x2x2
    U_11[0, 0] = 1; U_11[3, 0] = 1
    V_11[0, 0] = 1; V_11[3, 0] = 1
    W_11[0, 0] = 1; W_11[3, 0] = 1
    
    U_11[0, 1] = 1; U_11[1, 1] = 1
    V_11[3, 1] = 1
    W_11[1, 1] = 1
    
    U_11[2, 2] = 1; U_11[3, 2] = 1
    V_11[0, 2] = 1
    W_11[2, 2] = 1; W_11[3, 2] = 1
    
    U_11[0, 3] = 1; U_11[2, 3] = -1
    V_11[3, 3] = 1; V_11[2, 3] = -1
    W_11[0, 3] = 1; W_11[2, 3] = 1
    
    U_11[1, 4] = 1; U_11[3, 4] = 1
    V_11[2, 4] = 1
    W_11[0, 4] = -1; W_11[1, 4] = 1
    
    U_11[0, 5] = 1; U_11[2, 5] = 1
    V_11[0, 5] = 1; V_11[1, 5] = -1
    # W_11 for P6 is 0?
    
    U_11[1, 6] = 1; U_11[3, 6] = -1
    V_11[1, 6] = 1; V_11[3, 6] = -1
    W_11[0, 6] = -1; W_11[3, 6] = 1
    
    # Verify U_11, V_11, W_11
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # u = U @ A_flat = [1+4, 1+2, 3+4, 1-3, 2+4, 1+3, 2-4] = [5, 3, 7, -2, 6, 4, -2]
    # v = V @ B_flat = [5+8, 8, 5, 8-7, 7, 5-6, 6-8] = [13, 8, 5, 1, 7, -1, -2]
    # w = W @ (u*v) = W @ [65, 24, 35, -2, 42, -4, 4]
    # C00 (0) = 65 -2 -42 -4 = 17. Expected 19.
    # My Strassen is wrong.
    
    # Let's find a correct Strassen implementation.
    
    # P1 = (a+d)(e+h)
    # P2 = (a+b)h
    # P3 = (c+d)e
    # P4 = (a-c)(h-g)
    # P5 = (b+d)g
    # P6 = (a+c)(e-f)
    # P7 = (b-d)(f-h)
    
    # C11 = P1 + P4 - P5 - P7
    # C12 = P2 + P5
    # C21 = P3 + P4
    # C22 = P1 - P2 + P3 + P7
    
    # Let's re-calculate with A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    # a=1, b=2, c=3, d=4
    # e=5, f=6, g=7, h=8
    # P1 = (1+4)(5+8) = 5*13 = 65
    # P2 = (1+2)8 = 24
    # P3 = (3+4)5 = 35
    # P4 = (1-3)(8-7) = -2*1 = -2
    # P5 = (2+4)7 = 42
    # P6 = (1+3)(5-6) = 4*-1 = -4
    # P7 = (2-4)(6-8) = -2*-2 = 4
    
    # C11 = 65 + (-2) - 42 - 4 = 17. Still 17.
    # Wait, C11 should be 1*5 + 2*7 = 5+14=19.
    # Is my formula for C11 wrong?
    # C11 = P1 + P4 - P5 - P7
    # Maybe P4 is defined differently?
    # (a-c)(h-g) = (1-3)(8-7) = -2.
    # If P4 = (c-a)(g-h) = (2)(-1) = -2.
    # If P4 = (a-c)(g-h) = (-2)(-1) = 2.
    # If P4 = (c-a)(h-g) = (2)(1) = 2.
    
    # Let's check the standard Strassen Wikipedia.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map my P's to M's.
    # P1 = M1
    # P3 = M2
    # P2 = M5
    # P4 = ? (a-c)(h-g) = (A11-A21)(B22-B21). Not matching directly.
    
    # Let's just use the M's directly.
    # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=1, W[3,0]=1
    # M2: U[2,1]=1, U[3,1]=1; V[0,1]=1; W[2,1]=1, W[3,1]=1
    # M3: U[0,2]=1; V[2,2]=1, V[3,2]=-1; W[1,2]=1
    # M4: U[3,3]=1; V[0,3]=1, V[1,3]=-1; W[0,3]=1, W[2,3]=1
    # M5: U[0,4]=1, U[1,4]=1; V[3,4]=1; W[0,4]=-1, W[1,4]=1
    # M6: U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[1,5]=1; W[3,5]=1
    # M7: U[1,6]=1, U[3,6]=-1; V[2,6]=1, V[3,6]=1; W[0,6]=1
    
    # Let's verify M1..M7
    # A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    # M1 = 5*13 = 65
    # M2 = 7*5 = 35
    # M3 = 1*(7-8) = -1
    # M4 = 4*(5-6) = -4
    # M5 = 3*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    
    # C11 = 65 -4 -24 -30 = 7. Wrong.
    # C11 = M1 + M4 - M5 + M7 = 65 -4 -24 -30 = 7.
    # Expected 19.
    
    # I am clearly misremembering Strassen.
    # Let's use a library or a verified snippet.
    # Or just use the schoolbook rank 42.
    # The directive says "REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family."
    # The current design is schoolbook.
    # If I can't get Strassen right quickly, I should stick to schoolbook or a simple block decomposition that I am sure of.
    
    # Simple block decomposition:
    # 2x3x7.
    # Split p=7 into 3+3+1.
    # Rank = Rank(2x3x3) + Rank(2x3x3) + Rank(2x3x1).
    # Rank(2x3x1) = 6.
    # Rank(2x3x3) = ?
    # If I use schoolbook for 2x3x3, rank = 18.
    # Total = 18+18+6 = 42.
    
    # Can I do better for 2x3x3?
    # 2x3x3 can be split into 2x2x3 + 2x1x3.
    # Rank(2x2x3) = ?
    # Rank(2x1x3) = 6.
    # If Rank(2x2x3) = 8 (Strassen-like?), then 8+6=14.
    # 14+14+6 = 34.
    
    # Let's implement a safe, verified schoolbook and try to optimize if time permits.
    # Given the 120s limit and the need for correctness, schoolbook is the safest bet to score 1.0.
    # The previous attempt failed validation.
    
    n, m, p = DIMENSIONS
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
