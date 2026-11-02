"""Optimized Rank-22 decomposition for 2x3x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # A indices:
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    
    # B indices:
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # B20=8, B21=9, B22=10, B23=11
    
    # C indices (k*n+i):
    # C00=0, C01=1
    # C10=2, C11=3
    # C20=4, C21=5
    # C30=6, C31=7
    
    col = 0
    
    # Strassen for A_LL * B_LL
    # M1: (A00+A11)(B00+B11)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    # M2: (A10+A11)B01
    U[3, col] = 1; U[4, col] = 1
    V[1, col] = 1
    W[1, col] = 1
    col += 1
    # M3: A00(B10-B11)
    U[0, col] = 1
    V[4, col] = 1; V[5, col] = -1
    W[2, col] = 1
    col += 1
    # M4: A11(B01-B00)
    U[4, col] = 1
    V[1, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    # M5: (A00+A01)B11
    U[0, col] = 1; U[1, col] = 1
    V[5, col] = 1
    W[0, col] = -1; W[1, col] = 1
    col += 1
    # M6: (A10-A00)(B00+B01)
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[2, col] = 1
    col += 1
    # M7: (A01-A11)(B10+B11)
    U[1, col] = 1; U[4, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[3, col] = -1
    col += 1
    
    # Outer product a * b_L
    # a = [A02, A12]
    # b_L = [B20, B21]
    # M8: A02 B20
    U[2, col] = 1
    V[8, col] = 1
    W[0, col] = 1
    col += 1
    # M9: A12 B21
    U[5, col] = 1
    V[9, col] = 1
    W[3, col] = 1
    col += 1
    # M10: (A02+A12)(B20+B21)
    U[2, col] = 1; U[5, col] = 1
    V[8, col] = 1; V[9, col] = 1
    W[1, col] = 1; W[2, col] = 1
    col += 1
    # M11: A12 B20
    U[5, col] = 1
    V[8, col] = 1
    W[2, col] = 1
    col += 1
    # A02 B21 = M10 - M8 - M9 - M11
    # Add to C01 (index 1)
    # W[1] += M10 - M8 - M9 - M11
    # We already added M10 to W[1].
    # Need to subtract M8, M9, M11 from W[1].
    # But we can't go back.
    # We must add the negative terms to W[1] in the current or future columns?
    # No, we can just add the term A02 B21 directly?
    # If we use 4 mults, we don't need to derive it.
    # M8: A02 B20
    # M9: A12 B21
    # M10: A02 B21
    # M11: A12 B20
    # This is simpler.
    
    # Let's restart the Outer Product part with 4 explicit mults.
    # col is 8.
    # M8: A02 B20
    U[2, col] = 1
    V[8, col] = 1
    W[0, col] = 1
    col += 1
    # M9: A12 B21
    U[5, col] = 1
    V[9, col] = 1
    W[3, col] = 1
    col += 1
    # M10: A02 B21
    U[2, col] = 1
    V[9, col] = 1
    W[1, col] = 1
    col += 1
    # M11: A12 B20
    U[5, col] = 1
    V[8, col] = 1
    W[2, col] = 1
    col += 1
    
    # Strassen for A_LL * B_RL
    # B_RL: B02, B03, B12, B13
    # M12: (A00+A11)(B02+B13)
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[7, col] = 1
    W[4, col] = 1; W[7, col] = 1
    col += 1
    # M13: (A10+A11)B03
    U[3, col] = 1; U[4, col] = 1
    V[3, col] = 1
    W[5, col] = 1
    col += 1
    # M14: A00(B12-B13)
    U[0, col] = 1
    V[6, col] = 1; V[7, col] = -1
    W[6, col] = 1
    col += 1
    # M15: A11(B03-B02)
    U[4, col] = 1
    V[3, col] = 1; V[2, col] = -1
    W[4, col] = 1; W[7, col] = 1
    col += 1
    # M16: (A00+A01)B13
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1
    W[4, col] = -1; W[5, col] = 1
    col += 1
    # M17: (A10-A00)(B02+B03)
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[6, col] = 1
    col += 1
    # M18: (A01-A11)(B12+B13)
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[4, col] = 1; W[7, col] = -1
    col += 1
    
    # Outer product a * b_R
    # a = [A02, A12]
    # b_R = [B22, B23]
    # M19: A02 B22
    U[2, col] = 1
    V[10, col] = 1
    W[4, col] = 1
    col += 1
    # M20: A12 B23
    U[5, col] = 1
    V[11, col] = 1
    W[7, col] = 1
    col += 1
    # M21: A02 B23
    U[2, col] = 1
    V[11, col] = 1
    W[5, col] = 1
    col += 1
    # M22: A12 B22
    U[5, col] = 1
    V[10, col] = 1
    W[6, col] = 1
    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
