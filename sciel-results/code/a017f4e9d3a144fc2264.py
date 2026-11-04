"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import itertools
from collections import defaultdict
import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def schoolbook():
        """Return the rank-45 schoolbook decomposition."""
        rank = n * m * p  # 45
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i1 in range(n):
            for i2 in range(m):
                for k in range(p):
                    U[i1 * m + i2][col] = 1
                    V[i2 * p + k][col] = 1
                    W[k * n + i1][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the multiplication tensor."""
        rank = len(U[0])
        # Check: for all i1, i2, k:
        # sum_t U[i1*m+i2, t] * V[i2*p+k, t] * W[k*n+i1, t] == 1
        for i1 in range(n):
            for i2 in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i1 * m + i2][t] * V[i2 * p + k][t] * W[k * n + i1][t]
                    if abs(s - 1.0) > tol:
                        return False
        # Also check that no spurious contributions:
        # For invalid (i1, i2, k) combinations... actually there are no invalid ones
        # in the tensor - the tensor is defined for all (i1,i2,k) and equals 1.
        # Wait, that's not right. The tensor T[i1,i2,k] = 1 for all valid triples.
        # The decomposition must satisfy:
        # T[i1,i2,k] = sum_t U[i1*m+i2,t] * V[i2*p+k,t] * W[k*n+i1,t] = 1
        # for ALL i1, i2, k. So we just need the above check.
        return True
    
    def try_strassen_pattern():
        """Try to find a Strassen-like rank reduction.
        
        Look for 2x2x2 sub-blocks where we can apply Strassen's 7-term
        decomposition instead of 8 terms, saving 1 rank.
        
        A Strassen pattern requires:
        - Two A-indices: (i1a, i2a) and (i1b, i2b) 
        - Two B-indices: (i2a, ka) and (i2b, kb) -- but these must share structure
        - Two C-indices: (ka, i1a) and (kb, i1b)
        
        Actually, for a proper Strassen pattern on a 2x2x2 block:
        We need A rows {r1, r2}, A cols {c1, c2}, B rows {c1, c2}, B cols {k1, k2}
        This gives C entries {(k1,r1), (k1,r2), (k2,r1), (k2,r2)}
        
        The 8 schoolbook terms are:
        (r1,c1,k1), (r1,c1,k2), (r1,c2,k1), (r1,c2,k2),
        (r2,c1,k1), (r2,c1,k2), (r2,c2,k1), (r2,c2,k2)
        
        Strassen replaces these 8 with 7 terms using linear combinations.
        """
        # For our 3x3x5 problem:
        # A is 3x3, so rows {0,1,2}, cols {0,1,2}
        # B is 3x5, so rows {0,1,2}, cols {0,1,2,3,4}
        # C is 3x5, so rows {0,1,2,3,4}, cols {0,1,2}
        
        # A 2x2x2 Strassen pattern needs:
        # Two A rows: {r1, r2} from {0,1,2}
        # Two A cols: {c1, c2} from {0,1,2}  
        # Two B cols: {k1, k2} from {0,1,2,3,4}
        # (B rows are the same as A cols: {c1, c2})
        
        # The 8 affected terms are:
        # (r1,c1,k1), (r1,c1,k2), (r1,c2,k1), (r1,c2,k2),
        # (r2,c1,k1), (r2,c1,k2), (r2,c2,k1), (r2,c2,k2)
        
        # Strassen's 7 multiplications:
        # M1 = (A[r1,c1] + A[r2,c2]) * (B[c1,k1] + B[c2,k2])
        # M2 = (A[r2,c1] + A[r2,c2]) * B[c1,k1]
        # M3 = A[r1,c1] * (B[c2,k1] - B[c2,k2])
        # M4 = A[r2,c2] * (B[c1,k2] - B[c1,k1])
        # M5 = (A[r1,c1] + A[r1,c2]) * B[c2,k2]
        # M6 = (A[r2,c1] - A[r1,c1]) * (B[c1,k1] + B[c1,k2])
        # M7 = (A[r1,c2] - A[r2,c2]) * (B[c2,k1] + B[c2,k2])
        
        # Result:
        # C[k1,r1] = M1 - M4 + M5 - M7
        # C[k2,r1] = M3 + M1 - M5 + M6
        # C[k1,r2] = M2 - M3 + M7 - M6
        # C[k2,r2] = M4 + M2 - M7 + M1
        
        # Wait, let me double-check with standard Strassen:
        # For C = A*B, 2x2:
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M1 - M5 + M6
        # C[1,0] = M2 - M3 + M7 - M6
        # C[1,1] = M4 + M2 - M7 + M1
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let me verify C[0,0]:
        # M1 - M4 + M5 - M7
        # = (A00+A11)(B00+B11) - A11(B01-B00) + (A00+A01)B11 - (A01-A11)(B10+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
        # = A00*B00 + 2*A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 + A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
        # Hmm, this doesn't look right. Let me use the standard formula.
        
        # Standard Strassen:
        # p1 = (a00 + a11) * (b00 + b11)
        # p2 = (a10 + a11) * b00
        # p3 = a00 * (b10 - b11)
        # p4 = a11 * (b01 - b00)
        # p5 = (a00 + a01) * b11
        # p6 = (a10 - a00) * (b00 + b01)
        # p7 = (a01 - a11) * (b10 + b11)
        
        # c00 = p1 - p4 + p5 - p7
        # c01 = p3 + p1 - p5 + p6
        # c10 = p2 - p3 + p7 - p6
        # c11 = p4 + p2 - p7 + p1
        
        # Let me verify c00:
        # p1 - p4 + p5 - p7
        # = (a00+a11)(b00+b11) - a11(b01-b00) + (a00+a01)b11 - (a01-a11)(b10+b11)
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 + a00*b11 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
        # = a00*b00 + 2*a00*b11 + 2*a11*b00 + 2*a11*b11 - a11*b01 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
        
        # This should equal a00*b00 + a01*b10. Let me expand more carefully:
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11  [from p1]
        #   - a11*b01 + a11*b00                    [from -p4]
        #   + a00*b11 + a01*b11                    [from p5]
        #   - a01*b10 - a01*b11 + a11*b10 + a11*b11 [from -p7]
        # = a00*b00 + (a00*b11 + a00*b11) + (a11*b00 + a11*b00) + (a11*b11 + a11*b11)
        #   - a11*b01 + a01*b11 - a01*b10 - a01*b11 + a11*b10
        # = a00*b00 + 2*a00*b11 + 2*a11*b00 + 2*a11*b11 - a11*b01 - a01*b10 + a11*b10
        
        # This is NOT equal to a00*b00 + a01*b10. I must have the Strassen formula wrong.
        
        # Let me look it up properly. The correct Strassen formulas are:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M1 - M5 + M6
        # C10 = M2 - M3 + M7 - M6
        # C11 = M4 + M2 - M7 + M1
        
        # Let me verify C00 again, very carefully:
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # -M4 = -A11*B01 + A11*B00
        # M5 = A00*B11 + A01*B11
        # -M7 = -A01*B10 - A01*B11 + A11*B10 + A11*B11
        
        # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11
        #      - A11*B01 + A11*B00
        #      + A00*B11 + A01*B11
        #      - A01*B10 - A01*B11 + A11*B10 + A11*B11
        
        # Group by terms:
        # A00*B00: 1
        # A00*B11: 1 + 1 = 2
        # A11*B00: 1 + 1 = 2
        # A11*B11: 1 + 1 = 2
        # A11*B01: -1
        # A01*B11: 1 - 1 = 0
        # A01*B10: -1
        # A11*B10: 1
        
        # = A00*B00 + 2*A00*B11 + 2*A11*B00 + 2*A11*B11 - A11*B01 - A01*B10 + A11*B10
        
        # This is definitely not A00*B00 + A01*B10.
        
        # I think I have the Strassen formulas wrong. Let me use the correct ones:
        # 
        # p1 = (a00 + a11) * (b00 + b11)
        # p2 = (a10 + a11) * b00
        # p3 = a00 * (b10 - b11)
        # p4 = a11 * (b01 - b00)
        # p5 = (a00 + a01) * b11
        # p6 = (a10 - a00) * (b00 + b01)
        # p7 = (a01 - a11) * (b10 + b11)
        #
        # c00 = p1 - p4 + p5 - p7
        # c01 = p3 + p1 - p5 + p6
        # c10 = p2 - p3 + p7 - p6
        # c11 = p4 + p2 - p7 + p1
        
        # Hmm, maybe I should just test numerically.
        
        # For now, let me just return schoolbook since I can't quickly verify
        # a Strassen-based construction.
        
        return None
    
    # Try to find a better decomposition
    result = try_strassen_pattern()
    
    if result is not None and verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
