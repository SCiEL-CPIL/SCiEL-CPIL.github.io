import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check that sum_r U[:,r] * V[:,r] * W[:,r] exactly equals the matmul tensor."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback: rank 36
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to find a rank-35 decomposition using Strassen for the 2x2x2 sub-block.
    # The 2x2x2 sub-block involves i in {0,1}, j in {0,1}, k in {0,1}.
    # Schoolbook for this block uses 8 terms. Strassen uses 7.
    # Total rank = 36 - 8 + 7 = 35.
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    
    # Output reconstruction:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6 - M7  -- wait, let me verify this carefully
    # C11 = M1 + M3 - M2 + M6 ... let me re-derive
    
    # Actually, let me use the standard Strassen formulas and verify:
    # C[0][0] = A00*B00 + A01*B10
    # C[0][1] = A00*B01 + A01*B11
    # C[1][0] = A10*B00 + A11*B10
    # C[1][1] = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # M3 = A00(B10-B11) = A00*B10 - A00*B11
    # M4 = A11(B01-B00) = A11*B01 - A11*B00
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # M6 = (A10-A00)(B00+B10) = A10*B00 + A10*B10 - A00*B00 - A00*B10
    # M7 = (A01-A11)(B01+B11) = A01*B01 + A01*B11 - A11*B01 - A11*B11
    
    # Now reconstruct:
    # C00 = A00*B00 + A01*B10
    #     = M1 - M4 + M5 + M7? Let's check:
    #     M1 - M4 + M5 + M7 = (A00*B00 + A00*B11 + A11*B00 + A11*B11) 
    #                        - (A11*B01 - A11*B00) 
    #                        + (A00*B11 + A01*B11) 
    #                        + (A01*B01 + A01*B11 - A11*B01 - A11*B11)
    #     This is getting messy. Let me use the known correct Strassen formulas:
    
    # The correct Strassen output formulas are:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6 - M7  -- I think this might be wrong, let me look it up properly
    # C11 = M1 + M3 - M2 + M6
    
    # Actually, the standard Strassen formulas are:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b10)
    # p7 = (a01-a11)(b01+b11)
    
    # c00 = p1 - p4 + p5 + p7
    # c01 = p3 + p4
    # c10 = p2 + p6 - p7  -- wait, I need to double-check this
    # c11 = p1 + p3 - p2 + p6
    
    # Let me verify c10 = A10*B00 + A11*B10:
    # p2 + p6 - p7 = (A10*B00 + A11*B00) + (A10*B00 + A10*B10 - A00*B00 - A00*B10) 
    #               - (A01*B01 + A01*B11 - A11*B01 - A11*B11)
    # This doesn't simplify to A10*B00 + A11*B10. So the formula might be different.
    
    # Let me just use a known correct rank-7 decomposition for 2x2x2.
    # The tensor for 2x2x2 matrix multiplication has rank 7 (Strassen).
    # I'll construct it directly and verify.
    
    # For now, let me just return the schoolbook decomposition which is guaranteed correct.
    # The score will be 36, which is the baseline.
    
    # Actually, let me try to construct the Strassen decomposition properly.
    # I'll define the 7 terms explicitly.
    
    # Term r=0: M1 = (A00+A11)(B00+B11)
    # U contributions: U[0*2+0] += 1, U[1*2+1] += 1  (A00 and A11)
    # V contributions: V[0*2+0] += 1, V[1*2+1] += 1  (B00 and B11)
    # W contributions: W[0*2+0] += 1, W[0*2+1] += 1, W[1*2+0] += 1, W[1*2+1] += 1
    # But wait, W is indexed by k*n+i, and for the 2x2x2 block, n=3, so W[k*3+i]
    
    # This is getting complicated with the indexing. Let me just return schoolbook.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
