"""Matrix multiplication 2x3x6 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # We decompose into 3 blocks of 2 columns each (k pairs: (0,1), (2,3), (4,5))
    # For each block, we use Strassen-like decomposition for the 2x2 part (j=0,1)
    # and direct multiplication for the 2x1 part (j=2)
    
    # Strassen for 2x2x2:
    # C00 = P1 - P3 - P4 + P7
    # C01 = P5 + P4
    # C10 = P1 + P2
    # C11 = P1 - P2 + P3 + P5
    # where:
    # P1 = A00*(B00-B11)
    # P2 = (A00+A01)*B11
    # P3 = (A10+A11)*B00
    # P4 = A11*(B10-B00)
    # P5 = (A00+A10)*B01
    # P6 = (A01-A10)*(B10+B11)
    # P7 = (A11-A01)*(B00+B01)
    
    # For our case, each block has:
    # - Strassen terms for j=0,1 (7 terms)
    # - Direct terms for j=2 (4 terms: A02*B2k0, A02*B2k1, A12*B2k0, A12*B2k1)
    # Total: 11 terms per block, 33 total
    
    # Let's define the indices
    # U index: i*m + j = i*3 + j, i in {0,1}, j in {0,1,2}
    # V index: j*p + k = j*6 + k, j in {0,1,2}, k in {0..5}
    # W index: k*n + i = k*2 + i, k in {0..5}, i in {0,1}
    
    col = 0
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen terms for j=0,1
        # P1 = A00*(B0k0-B1k1)
        U[0*3+0][col] = 1  # A00
        U[1*3+0][col] = 0  # A10
        U[0*3+1][col] = 0  # A01
        U[1*3+1][col] = 0  # A11
        V[0*6+k0][col] = 1  # B0k0
        V[1*6+k1][col] = -1  # -B1k1
        W[k0*2+0][col] = 1  # contributes to C0k0
        W[k0*2+1][col] = 0
        W[k1*2+0][col] = 0
        W[k1*2+1][col] = 0
        col += 1
        
        # P2 = (A00+A01)*B1k1
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*6+k1][col] = 1
        W[k0*2+0][col] = -1  # -P2 contributes to C00? Wait, let me recheck
        # Actually: C00 = P1 - P3 - P4 + P7, C01 = P5 + P4, C10 = P1 + P2, C11 = P1 - P2 + P3 + P5
        # So P2 contributes +1 to C10, -1 to C11
        W[k0*2+0][col] = 0
        W[k0*2+1][col] = 0
        W[k1*2+0][col] = 0
        W[k1*2+1][col] = 0
        # P2 should contribute to C10 and C11
        # Let me redo this more carefully
        col -= 1  # undo and redo
        
        # Let me be more systematic. For each Strassen term, I need to specify
        # which U entries, V entries, and W entries are non-zero.
        
        # P1 = A00 * (B0k0 - B1k1)
        # Contributes: +1 to C0k0, +1 to C1k0, -1 to C1k1
        U[0*3+0][col] = 1  # A00
        V[0*6+k0][col] = 1  # B0k0
        V[1*6+k1][col] = -1  # -B1k1
        W[k0*2+0][col] = 1  # C0k0
        W[k0*2+1][col] = 1  # C1k0
        W[k1*2+1][col] = -1  # -C1k1
        col += 1
        
        # P2 = (A00+A01) * B1k1
        # Contributes: +1 to C1k0, -1 to C1k1
        U[0*3+0][col] = 1  # A00
        U[0*3+1][col] = 1  # A01
        V[1*6+k1][col] = 1  # B1k1
        W[k0*2+1][col] = 1  # C1k0
        W[k1*2+1][col] = -1  # -C1k1
        col += 1
        
        # P3 = (A10+A11) * B0k0
        # Contributes: -1 to C0k0, +1 to C1k1
        U[1*3+0][col] = 1  # A10
        U[1*3+1][col] = 1  # A11
        V[0*6+k0][col] = 1  # B0k0
        W[k0*2+0][col] = -1  # -C0k0
        W[k1*2+1][col] = 1  # C1k1
        col += 1
        
        # P4 = A11 * (B1k0 - B0k0)
        # Contributes: -1 to C0k0, +1 to C0k1
        U[1*3+1][col] = 1  # A11
        V[1*6+k0][col] = 1  # B1k0
        V[0*6+k0][col] = -1  # -B0k0
        W[k0*2+0][col] = -1  # -C0k0
        W[k1*2+0][col] = 1  # C0k1
        col += 1
        
        # P5 = (A00+A10) * B0k1
        # Contributes: +1 to C0k1, +1 to C1k1
        U[0*3+0][col] = 1  # A00
        U[1*3+0][col] = 1  # A10
        V[0*6+k1][col] = 1  # B0k1
        W[k1*2+0][col] = 1  # C0k1
        W[k1*2+1][col] = 1  # C1k1
        col += 1
        
        # P6 = (A01-A10) * (B1k0+B1k1)
        # Contributes: +1 to C0k1? Let me check Strassen again
        # Actually P6 is not used in the standard 7-term Strassen for 2x2
        # The 7 terms are P1-P7, and P6 = (A01-A10)*(B10+B11)
        # P6 contributes to... actually in standard Strassen, P6 is not used for 2x2x2
        # Wait, I'm confusing myself. Let me use the correct Strassen decomposition.
        
        # Standard Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me restart with this correct formulation.
        
        # For block (k0, k1):
        # M1 = (A00+A11)*(B0k0+B1k1)
        # M2 = (A10+A11)*B0k0
        # M3 = A00*(B0k1-B1k1)
        # M4 = A11*(B1k0-B0k0)
        # M5 = (A00+A01)*B1k1
        # M6 = (A10-A00)*(B0k0+B0k1)
        # M7 = (A01-A11)*(B1k0+B1k1)
        
        # C0k0 = M1 - M4 + M5 - M7
        # C0k1 = M3 + M5
        # C1k0 = M2 + M4
        # C1k1 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)*(B0k0+B1k1)
        U[0*3+0][col] = 1  # A00
        U[1*3+1][col] = 1  # A11
        V[0*6+k0][col] = 1  # B0k0
        V[1*6+k1][col] = 1  # B1k1
        W[k0*2+0][col] = 1  # +C0k0
        W[k1*2+1][col] = 1  # +C1k1
        col += 1
        
        # M2 = (A10+A11)*B0k0
        U[1*3+0][col] = 1  # A10
        U[1*3+1][col] = 1  # A11
        V[0*6+k0][col] = 1  # B0k0
        W[k1*2+1][col] = -1  # -C1k1
        W[k0*2+1][col] = 1  # +C1k0
        col += 1
        
        # M3 = A00*(B0k1-B1k1)
        U[0*3+0][col] = 1  # A00
        V[0*6+k1][col] = 1  # B0k1
        V[1*6+k1][col] = -1  # -B1k1
        W[k0*2+1][col] = 1  # +C1k1
        W[k1*2+0][col] = 1  # +C0k1
        col += 1
        
        # M4 = A11*(B1k0-B0k0)
        U[1*3+1][col] = 1  # A11
        V[1*6+k0][col] = 1  # B1k0
        V[0*6+k0][col] = -1  # -B0k0
        W[k0*2+0][col] = -1  # -C0k0
        W[k0*2+1][col] = 1  # +C1k0
        col += 1
        
        # M5 = (A00+A01)*B1k1
        U[0*3+0][col] = 1  # A00
        U[0*3+1][col] = 1  # A01
        V[1*6+k1][col] = 1  # B1k1
        W[k0*2+0][col] = 1  # +C0k0
        W[k1*2+0][col] = 1  # +C0k1
        col += 1
        
        # M6 = (A10-A00)*(B0k0+B0k1)
        U[1*3+0][col] = 1  # A10
        U[0*3+0][col] = -1  # -A00
        V[0*6+k0][col] = 1  # B0k0
        V[0*6+k1][col] = 1  # B0k1
        W[k1*2+1][col] = 1  # +C1k1
        col += 1
        
        # M7 = (A01-A11)*(B1k0+B1k1)
        U[0*3+1][col] = 1  # A01
        U[1*3+1][col] = -1  # -A11
        V[1*6+k0][col] = 1  # B1k0
        V[1*6+k1][col] = 1  # B1k1
        W[k0*2+0][col] = -1  # -C0k0
        col += 1
        
        # Now direct terms for j=2
        # D1 = A02*B2k0 -> C0k0
        U[0*3+2][col] = 1  # A02
        V[2*6+k0][col] = 1  # B2k0
        W[k0*2+0][col] = 1  # C0k0
        col += 1
        
        # D2 = A02*B2k1 -> C0k1
        U[0*3+2][col] = 1  # A02
        V[2*6+k1][col] = 1  # B2k1
        W[k1*2+0][col] = 1  # C0k1
        col += 1
        
        # D3 = A12*B2k0 -> C1k0
        U[1*3+2][col] = 1  # A12
        V[2*6+k0][col] = 1  # B2k0
        W[k0*2+1][col] = 1  # C1k0
        col += 1
        
        # D4 = A12*B2k1 -> C1k1
        U[1*3+2][col] = 1  # A12
        V[2*6+k1][col] = 1  # B2k1
        W[k1*2+1][col] = 1  # C1k1
        col += 1
    
    return {"U": U, "V": V, "W": W}
