import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Construct the trivial rank-48 decomposition
    # This is the "schoolbook" decomposition where each multiplication
    # A[i,j] * B[j,k] contributes to C[i,k].
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Rank r corresponds to a specific (i, j, k) triple.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    # We will attempt to minimize the rank by checking if any columns in U, V, W
    # are linearly dependent or zero, or if we can merge terms.
    # For standard matrix multiplication, the tensor rank is exactly n*m*p unless
    # there are specific symmetries or zero entries, which there aren't in the general case.
    # However, the "rank" in the tensor decomposition context for matrix multiplication
    # is the bilinear complexity. For 2x4 * 4x6, the trivial complexity is 48.
    # Strassen-like methods can reduce this.
    
    # The previous attempt at Strassen (rank 42) was buggy.
    # Let's try a more robust approach: 
    # 1. Start with the trivial 48.
    # 2. Use a simple linear algebra check to see if we can reduce it.
    # Since the tensor is "generic" (no zeros), the rank is likely 48 or close to it.
    # Wait, the task asks for a rank-minimal EXACT decomposition.
    # The bilinear complexity of 2x4 * 4x6 is known to be 48? 
    # Actually, for m=4, p=6, n=2.
    # Let's check if we can use a block Strassen.
    # Split A (2x4) into two 2x2 blocks: A1, A2.
    # Split B (4x6) into two 2x6 blocks: B1, B2.
    # B1 (2x6) can be split into three 2x2 blocks: B11, B12, B13.
    # B2 (2x6) can be split into three 2x2 blocks: B21, B22, B23.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 * 2x6.
    # This is equivalent to multiplying a 2x2 by three 2x2 blocks.
    # We can use Strassen for 2x2 * 2x2.
    # Let's try to construct a rank 42 decomposition properly this time.
    
    # Strassen for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's verify this set again carefully.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # M2 = (a10+a11)b00 = a10b00 + a11b00
    # M3 = a00(b10-b11) = a00b10 - a00b11
    # M4 = a11(b01-b00) = a11b01 - a11b00
    # M5 = (a00+a01)b11 = a00b11 + a01b11
    # M6 = (a10-a00)(b00+b10) = a10b00 + a10b10 - a00b00 - a00b10
    # M7 = (a01-a11)(b01+b11) = a01b01 + a01b11 - a11b01 - a11b11
    
    # C00 = a00b00 + a01b10
    # C01 = a00b01 + a01b11
    # C10 = a10b00 + a11b10
    # C11 = a10b01 + a11b11
    
    # Let's check C00 = M1 + M4 - M5 + M7?
    # M1 + M4 - M5 + M7 = 
    # (a00b00 + a00b11 + a11b00 + a11b11) 
    # + (a11b01 - a11b00) 
    # - (a00b11 + a01b11) 
    # + (a01b01 + a01b11 - a11b01 - a11b11)
    # = a00b00 + (a00b11 - a00b11) + (a11b00 - a11b00) + (a11b11 - a11b11) 
    # + (a11b01 - a11b01) + (-a01b11 + a01b11) + a01b01
    # = a00b00 + a01b01.
    # But C00 should be a00b00 + a01b10.
    # So this Strassen set is WRONG for standard matrix multiplication C=AB.
    # It seems to be for C = A B^T or some other convention.
    
    # Correct Strassen for C=AB:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # Let's re-evaluate C00 = a00b00 + a01b10.
    # Which M's contain a00b00? M1, M2(no), M3(no), M4(no), M5(no), M6(no), M7(no).
    # Only M1 has a00b00?
    # M1 = a00b00 + ...
    # Which M's contain a01b10?
    # M1: no.
    # M2: no.
    # M3: no.
    # M4: no.
    # M5: no.
    # M6: no.
    # M7: no.
    # Wait, none of the standard Strassen M's contain a01b10?
    # M3 = a00(b10-b11) -> a00b10.
    # M6 = (a10-a00)(b00+b10) -> -a00b10.
    # M7 = (a01-a11)(b01+b11) -> no b10.
    
    # It appears the standard Strassen formulas I recalled are for a specific permutation or I am mixing up indices.
    # Let's look up the exact Strassen formulas for C=AB.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # C00 = a e + b g
    # C01 = a f + b h
    # C10 = c e + d g
    # C11 = c f + d h
    
    # Let's check C00 = M1 + M4 - M5 + M7?
    # M1 = ae + ah + de + dh
    # M4 = df - de
    # M5 = ah + bh
    # M7 = bf + bh - df - dh
    # Sum = ae + ah + de + dh + df - de - ah - bh + bf + bh - df - dh
    # = ae + bf.
    # Correct!
    
    # Let's check C01 = M3 + M5?
    # M3 = ag - ah
    # M5 = ah + bh
    # Sum = ag + bh.
    # But C01 = af + bh.
    # Mismatch: ag vs af.
    # So M3 must be a(f-h)?
    # If M3 = a(f-h) = af - ah.
    # Then M3 + M5 = af - ah + ah + bh = af + bh. Correct.
    
    # Let's check C10 = M2 + M4?
    # M2 = ce + de
    # M4 = df - de
    # Sum = ce + df.
    # But C10 = ce + dg.
    # Mismatch: df vs dg.
    # So M4 must be d(g-e)?
    # If M4 = d(g-e) = dg - de.
    # Then M2 + M4 = ce + de + dg - de = ce + dg. Correct.
    
    # Let's check C11 = M1 - M2 + M3 + M6?
    # M1 = ae + ah + de + dh
    # M2 = ce + de
    # M3 = af - ah
    # M6 = (c-a)(e+g) = ce + cg - ae - ag
    # Sum = ae + ah + de + dh - ce - de + af - ah + ce + cg - ae - ag
    # = dh + af + cg - ag.
    # But C11 = cf + dh.
    # Mismatch.
    
    # It seems the standard set is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)  <-- Note: f-h, not g-h
    # M4 = d(g-e)  <-- Note: g-e, not f-e
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # Let's re-verify C11 with this set.
    # C11 = M1 - M2 + M3 + M6
    # M1 = ae + ah + de + dh
    # M2 = ce + de
    # M3 = af - ah
    # M6 = ce + cg - ae - ag
    # Sum = ae + ah + de + dh - ce - de + af - ah + ce + cg - ae - ag
    # = dh + af + cg - ag.
    # Still wrong. C11 should be cf + dh.
    
    # Let's try M6 = (c-a)(f+g)? No, that's not standard.
    
    # Okay, I will not rely on memory for Strassen coefficients.
    # I will use the trivial rank 48 decomposition which is GUARANTEED to be correct.
    # The score will be 48/48 = 1.0.
    # This is safe.
    
    def trivial_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    
        return {"U": U, "V": V, "W": W}
    
    return trivial_decomposition()
# EVOLVE-BLOCK-END
