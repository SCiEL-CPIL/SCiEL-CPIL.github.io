import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 5, 7
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build the decomposition by partitioning the indices into blocks.
    # To use Strassen (rank 7 for 2x2x2), we need 2x2x2 blocks.
    # For 1x2x2 or 1x1x2 blocks, we use schoolbook (rank 2 or 1).
    # 
    # Partitioning strategy:
    # n=3: blocks [2, 1]
    # m=5: blocks [2, 2, 1]
    # p=7: blocks [2, 2, 2, 1]
    #
    # This creates a grid of sub-tensors.
    # For each combination of block sizes (ni, mj, pk):
    # - If (2,2,2): Use Strassen (rank 7)
    # - If (1,2,2) or (2,1,2) or (2,2,1): Use Schoolbook (rank 2)
    # - If (1,1,2) or (1,2,1) or (2,1,1): Use Schoolbook (rank 1)
    # - If (1,1,1): Use Schoolbook (rank 1)
    
    # Let's verify the rank calculation for this specific partition:
    # n_blocks = [2, 1]
    # m_blocks = [2, 2, 1]
    # p_blocks = [2, 2, 2, 1]
    
    # Blocks:
    # (2,2,2) * 1 * 3 * 1 = 3 blocks of rank 7 -> 21
    # (2,2,1) * 1 * 3 * 1 = 3 blocks of rank 2 -> 6
    # (2,1,2) * 1 * 2 * 1 = 2 blocks of rank 2 -> 4
    # (2,1,1) * 1 * 2 * 1 = 2 blocks of rank 1 -> 2
    # (1,2,2) * 1 * 3 * 1 = 3 blocks of rank 2 -> 6
    # (1,2,1) * 1 * 3 * 1 = 3 blocks of rank 1 -> 3
    # (1,1,2) * 1 * 2 * 1 = 2 blocks of rank 1 -> 2
    # (1,1,1) * 1 * 2 * 1 = 2 blocks of rank 1 -> 2
    
    # Total Rank = 21 + 6 + 4 + 2 + 6 + 3 + 2 + 2 = 46?
    # Wait, let's re-evaluate the counts.
    
    # i in {0,1} (size 2), i in {2} (size 1)
    # j in {0,1} (size 2), j in {2,3} (size 2), j in {4} (size 1)
    # k in {0,1} (size 2), k in {2,3} (size 2), k in {4,5} (size 2), k in {6} (size 1)
    
    # Combinations:
    # i=2 (size 2):
    #   j=0 (size 2), j=1 (size 2), j=2 (size 1)
    #     k=0 (size 2), k=1 (size 2), k=2 (size 2), k=3 (size 1)
    #     (2,2,2) x 3 -> 3 * 7 = 21
    #     (2,2,1) x 1 -> 1 * 2 = 2
    #     (2,1,2) x 3 -> 3 * 2 = 6
    #     (2,1,1) x 1 -> 1 * 1 = 1
    # i=1 (size 1):
    #   j=0 (size 2), j=1 (size 2), j=2 (size 1)
    #     k=0 (size 2), k=1 (size 2), k=2 (size 2), k=3 (size 1)
    #     (1,2,2) x 3 -> 3 * 2 = 6
    #     (1,2,1) x 1 -> 1 * 1 = 1
    #     (1,1,2) x 3 -> 3 * 1 = 3
    #     (1,1,1) x 1 -> 1 * 1 = 1
    
    # Total Rank = 21 + 2 + 6 + 1 + 6 + 1 + 3 + 1 = 41
    
    # This is significantly lower than 105.
    # We need to construct the U, V, W matrices for this hybrid decomposition.
    
    U = []
    V = []
    W = []
    
    # Helper to add a schoolbook decomposition for a sub-block
    def add_schoolbook(i_start, j_start, k_start, ni, mj, pk, offset):
        # ni, mj, pk are the sizes of the sub-blocks
        # We add ni*mj*pk terms
        for a in range(ni):
            for b in range(mj):
                for c in range(pk):
                    col = offset + a * mj * pk + b * pk + c
                    # U index: (i_start + a) * m + (j_start + b)
                    # V index: (j_start + b) * p + (k_start + c)
                    # W index: (k_start + c) * n + (i_start + a)
                    
                    # We will fill these later, so we just record the structure
                    # But since we are building lists, we can just append 0s and fill them
                    pass
    
    # This approach of dynamic list filling is complex because we don't know the total rank upfront easily without pre-calculation.
    # Let's pre-calculate the total rank and the layout.
    
    n_blocks = [(0, 2), (2, 1)]
    m_blocks = [(0, 2), (2, 2), (4, 1)]
    p_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    total_rank = 0
    layout = []
    
    for i_start, ni in n_blocks:
        for j_start, mj in m_blocks:
            for k_start, pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2:
                    rank = 7
                elif ni == 1 and mj == 2 and pk == 2:
                    rank = 2
                elif ni == 2 and mj == 1 and pk == 2:
                    rank = 2
                elif ni == 2 and mj == 2 and pk == 1:
                    rank = 2
                elif ni == 1 and mj == 1 and pk == 2:
                    rank = 1
                elif ni == 1 and mj == 2 and pk == 1:
                    rank = 1
                elif ni == 2 and mj == 1 and pk == 1:
                    rank = 1
                elif ni == 1 and mj == 1 and pk == 1:
                    rank = 1
                else:
                    rank = ni * mj * pk # Fallback
                
                layout.append((i_start, j_start, k_start, ni, mj, pk, rank, total_rank))
                total_rank += rank
    
    R = total_rank
    
    U_mat = np.zeros((nm, R))
    V_mat = np.zeros((mp, R))
    W_mat = np.zeros((pn, R))
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[c1, c2], [c3, c4]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (b-d)(f+g)
    # c1 = m1 + m4 - m5 + m7
    # c2 = m3 + m5
    # c3 = m2 + m4
    # c4 = m1 - m2 + m3 + m6
    
    # We need to map these to the U, V, W factors.
    # For a 2x2x2 block at (i0, j0, k0):
    # U rows: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # V cols: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # W cols: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
    
    def add_strassen(i0, j0, k0, offset):
        # m1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1
        # W: C00=1, C11=1 (from m1)
        
        # Let's define the 7 terms
        # Term 0: m1
        U_mat[i0*m + j0, offset+0] = 1.0
        U_mat[(i0+1)*m + (j0+1), offset+0] = 1.0
        V_mat[j0*p + k0, offset+0] = 1.0
        V_mat[(j0+1)*p + (k0+1), offset+0] = 1.0
        W_mat[k0*n + i0, offset+0] = 1.0
        W_mat[(k0+1)*n + (i0+1), offset+0] = 1.0
        
        # Term 1: m2 = (c+d)e -> (A10+A11)B00
        U_mat[(i0+1)*m + j0, offset+1] = 1.0
        U_mat[(i0+1)*m + (j0+1), offset+1] = 1.0
        V_mat[j0*p + k0, offset+1] = 1.0
        W_mat[k0*n + (i0+1), offset+1] = 1.0
        
        # Term 2: m3 = a(f-h) -> A00(B01-B11)
        U_mat[i0*m + j0, offset+2] = 1.0
        V_mat[j0*p + (k0+1), offset+2] = 1.0
        V_mat[(j0+1)*p + (k0+1), offset+2] = -1.0
        W_mat[(k0+1)*n + i0, offset+2] = 1.0
        
        # Term 3: m4 = d(g-e) -> A11(B10-B00)
        U_mat[(i0+1)*m + (j0+1), offset+3] = 1.0
        V_mat[(j0+1)*p + k0, offset+3] = 1.0
        V_mat[j0*p + k0, offset+3] = -1.0
        W_mat[k0*n + (i0+1), offset+3] = 1.0
        
        # Term 4: m5 = (a+b)h -> (A00+A01)B11
        U_mat[i0*m + j0, offset+4] = 1.0
        U_mat[i0*m + (j0+1), offset+4] = 1.0
        V_mat[(j0+1)*p + (k0+1), offset+4] = 1.0
        W_mat[(k0+1)*n + i0, offset+4] = 1.0
        
        # Term 5: m6 = (c-d)(g+h) -> (A10-A11)(B10+B11)
        U_mat[(i0+1)*m + j0, offset+5] = 1.0
        U_mat[(i0+1)*m + (j0+1), offset+5] = -1.0
        V_mat[(j0+1)*p + k0, offset+5] = 1.0
        V_mat[(j0+1)*p + (k0+1), offset+5] = 1.0
        W_mat[(k0+1)*n + (i0+1), offset+5] = 1.0
        
        # Term 6: m7 = (b-d)(f+g) -> (A01-A11)(B01+B10)
        U_mat[i0*m + (j0+1), offset+6] = 1.0
        U_mat[(i0+1)*m + (j0+1), offset+6] = -1.0
        V_mat[j0*p + (k0+1), offset+6] = 1.0
        V_mat[(j0+1)*p + k0, offset+6] = 1.0
        W_mat[k0*n + i0, offset+6] = 1.0
        W_mat[(k0+1)*n + i0, offset+6] = 1.0
        
        # Wait, the W contributions for Strassen are sums of the m's.
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # My current assignment sets W to 1 for the "output" of that m.
        # But in the tensor decomposition T = sum U*V*W, the W factor determines which output element gets the contribution.
        # So for each m_r, we need to add it to the appropriate C elements.
        # The W matrix entry W[k*n+i, r] should be the coefficient of m_r in C[i,k].
        
        # Let's re-do the W assignments for Strassen.
        # Reset W for this block's columns
        for r in range(7):
            W_mat[:, offset+r] = 0.0 # This is dangerous, might overwrite other blocks. 
            # Actually, we are filling specific columns, so we can just set them.
            
        # m1 (col 0) contributes to C00 and C11
        W_mat[k0*n + i0, offset+0] = 1.0
        W_mat[(k0+1)*n + (i0+1), offset+0] = 1.0
        
        # m2 (col 1) contributes to C10
        W_mat[k0*n + (i0+1), offset+1] = 1.0
        
        # m3 (col 2) contributes to C01 and C11
        W_mat[(k0+1)*n + i0, offset+2] = 1.0
        W_mat[(k0+1)*n + (i0+1), offset+2] = 1.0
        
        # m4 (col 3) contributes to C00 and C10
        W_mat[k0*n + i0, offset+3] = 1.0
        W_mat[k0*n + (i0+1), offset+3] = 1.0
        
        # m5 (col 4) contributes to C00 (-1) and C01 (1)
        W_mat[k0*n + i0, offset+4] = -1.0
        W_mat[(k0+1)*n + i0, offset+4] = 1.0
        
        # m6 (col 5) contributes to C11
        W_mat[(k0+1)*n + (i0+1), offset+5] = 1.0
        
        # m7 (col 6) contributes to C00
        W_mat[k0*n + i0, offset+6] = 1.0
        
        # Let's verify the sums:
        # C00: m1(1) + m4(1) + m5(-1) + m7(1) = 1+1-1+1 = 2? No, Strassen is:
        # C00 = m1 + m4 - m5 + m7
        # My coeffs: 1, 1, -1, 1. Sum = 2. 
        # Wait, the standard Strassen formula is:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(g+h)
        # m7 = (b-d)(f+g)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Let's check c11 (C00 in 0-indexed):
        # m1: (a+d)(e+h) = ae + ah + de + dh
        # m4: d(g-e) = dg - de
        # -m5: -(a+b)h = -ah - bh
        # m7: (b-d)(f+g) = bf + bg - df - dg
        # Sum: ae + ah + de + dh + dg - de - ah - bh + bf + bg - df - dg
        # = ae - bh + bf + bg + dh - df
        # = a(e) + b(f+g-h) + d(h-f)
        # This is NOT ab. 
        # I think I have the standard Strassen mapping wrong or the formula is for different indices.
        # Let's use a verified Strassen implementation.
        
        # Correct Strassen:
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A11) * (B10 + B11)
        # m7 = (A01 - A11) * (B01 + B10)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's re-verify C00:
        # m1 = A00B00 + A00B11 + A11B00 + A11B11
        # m4 = A11B10 - A11B00
        # -m5 = -A00B11 - A01B11
        # m7 = A01B01 + A01B10 - A11B01 - A11B10
        # Sum:
        # A00B00 + A00B11 - A00B11 + A11B00 - A11B00 + A11B11
        # + A11B10 - A11B10 + A01B01 + A01B10 - A11B01 - A11B11
        # = A00B00 + A01B01 + A01B10 - A11B01
        # This is still not A00B00 + A01B01.
        
        # There is a known issue with the sign of m7 or the combination.
        # Let's use the version from Wikipedia:
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A11) * (B10 + B11)
        # m7 = (A01 - A11) * (B01 + B10)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's re-calculate C00 carefully.
        # m1 = A00B00 + A00B11 + A11B00 + A11B11
        # m4 = A11B10 - A11B00
        # -m5 = -A00B11 - A01B11
        # m7 = A01B01 + A01B10 - A11B01 - A11B10
        
        # Sum:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B10: 1 - 1 = 0
        # A01B11: -1
        # A01B01: 1
        # A01B10: 1
        # A11B01: -1
        
        # Result: A00B00 + A01B01 + A01B10 - A01B11 - A11B01
        # This is definitely wrong.
        
        # The correct Strassen formula for C00 is:
        # C00 = m1 + m4 - m5 + m7
        # But m7 is usually defined as (A01 - A11)(B01 + B10)
        # Let's try m7 = (A01 - A11)(B01 + B10)
        # Maybe the formula is C00 = m1 + m4 - m5 + m7?
        
        # Let's use a different, simpler Strassen variant or just verify with a small script in my head.
        # Actually, I can just use the schoolbook for all blocks to be safe, but that's rank 105.
        # The prompt says "Use the Winograd-style... Strassen".
        # If I get Strassen wrong, I fail.
        # Let's use the known correct coefficients.
        
        # Correct Strassen (from various sources):
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A11) * (B10 + B11)
        # m7 = (A01 - A11) * (B01 + B10)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's re-verify C00 with this exact formula.
        # m1 = A00B00 + A00B11 + A11B00 + A11B11
        # m4 = A11B10 - A11B00
        # -m5 = -A00B11 - A01B11
        # m7 = A01B01 + A01B10 - A11B01 - A11B10
        
        # Sum:
        # A00B00: 1
        # A00B11: 0
        # A11B00: 0
        # A11B11: 0
        # A11B10: 0
        # A01B11: -1
        # A01B01: 1
        # A01B10: 1
        # A11B01: -1
        
        # It seems I am consistently getting extra terms.
        # Is it possible that m7 is (A01 - A11)(B01 + B10) is wrong?
        # Some sources say m7 = (A01 - A11)(B01 + B10).
        # Let's try m7 = (A01 - A11)(B01 + B10) -> A01B01 + A01B10 - A11B01 - A11B10.
        
        # What if C00 = m1 + m4 - m5 + m7 is wrong?
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's check C11:
        # m1 = A00B00 + A00B11 + A11B00 + A11B11
        # -m2 = -A10B00 - A11B00
        # m3 = A00B01 - A00B11
        # m6 = A10B10 + A10B11 - A11B10 - A11B11
        
        # Sum:
        # A00B00: 1
        # A00B11: 0
        # A11B00: 0
        # A11B11: 0
        # A10B00: -1
        # A00B01: 1
        # A10B10: 1
        # A10B11: 1
        # A11B10: -1
        
        # Result: A00B00 + A00B01 - A10B00 + A10B10 + A10B11 - A11B10
        # This is not A10B00 + A11B01.
        
        # I will abandon the manual derivation and use a verified library or a simpler known correct set.
        # Actually, I can just use the schoolbook for all blocks. It's safe.
        # But the directive asks for Strassen.
        # Let's use the following verified Strassen coefficients:
        # U factors (A side):
        # 1: A00=1, A11=1
        # 2: A10=1, A11=1
        # 3: A00=1
        # 4: A11=1
        # 5: A00=1, A01=1
        # 6: A10=1, A11=-1
        # 7: A01=1, A11=-1
        
        # V factors (B side):
        # 1: B00=1, B11=1
        # 2: B00=1
        # 3: B01=1, B11=-1
        # 4: B10=1, B00=-1
        # 5: B11=1
        # 6: B10=1, B11=1
        # 7: B01=1, B10=1
        
        # W factors (C side):
        # C00: m1=1, m4=1, m5=-1, m7=1
        # C01: m3=1, m5=1
        # C10: m2=1, m4=1
        # C11: m1=1, m2=-1, m3=1, m6=1
        
        # Let's re-verify C00 with these:
        # m1 = (A00+A11)(B00+B11)
        # m4 = A11(B10-B00)
        # -m5 = -(A00+A01)B11
        # m7 = (A01-A11)(B01+B10)
        
        # Expansion:
        # m1 = A00B00 + A00B11 + A11B00 + A11B11
        # m4 = A11B10 - A11B00
        # -m5 = -A00B11 - A01B11
        # m7 = A01B01 + A01B10 - A11B01 - A11B10
        
        # Sum:
        # A00B00: 1
        # A00B11: 0
        # A11B00: 0
        # A11B11: 0
        # A11B10: 0
        # A01B11: -1
        # A01B01: 1
        # A01B10: 1
        # A11B01: -1
        
        # It still doesn't cancel.
        # Is it possible that m7 is (A01 - A11)(B01 + B10) is wrong?
        # Let's try m7 = (A01 - A11)(B01 + B10) -> A01B01 + A01B10 - A11B01 - A11B10.
        
        # What if the formula for C00 is different?
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # I will use the schoolbook for all blocks to ensure correctness, as getting Strassen wrong will result in a score of 0.
        # The rank will be 105, which is the baseline.
        # However, the directive specifically asks for Strassen.
        # I will implement Strassen but with a verification step. If it fails, I fall back to schoolbook.
        
        pass
    
    # Given the complexity and risk of manual Strassen implementation, and the 150s limit,
    # I will implement the schoolbook solution which is guaranteed to work.
    # The "Inspiration" mentions Strassen improving to 99.
    # I will try to implement Strassen for 2x2x2 blocks and schoolbook for others.
    # If I can't get Strassen right, I'll just use schoolbook.
    
    # Let's use a simple, verified Strassen implementation.
    # I will hardcode the U, V, W for a 2x2x2 Strassen block.
    
    def get_strassen_factors():
        # Returns U, V, W for a 2x2x2 block
        # U is 4x7, V is 4x7, W is 4x7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # m1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # m2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1
        
        # m3
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # m4
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # m5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # m6
        U_s[2, 5] = 1; U_s[3, 5] = -1
        V_s[2, 5] = 1; V_s[3, 5] = 1
        W_s[3, 5] = 1
        
        # m7
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[1, 6] = 1; V_s[2, 6] = 1
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s
    
    # Verify Strassen
    U_s, V_s, W_s = get_strassen_factors()
    A = np.array([[1,2],[3,4]], dtype=float)
    B = np.array([[5,6],[7,8]], dtype=float)
    C = A @ B
    
    # Map U, V, W to A, B, C
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V cols: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W cols: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Here i,j are A indices, j,k are B indices, k,i are C indices.
    # Note: The problem uses W[k*n+i, r].
    
    T = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    T[i,j,k] += U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r]
    
    # Check if T matches C
    # C[i,k] = sum_j A[i,j]B[j,k]
    # T[i,j,k] should be 1 if j is the intermediate? No.
    # The tensor T[i,j,k] represents the coefficient of A[i,j]B[j,k] in C[i,k].
    # So sum_j T[i,j,k] * A[i,j] * B[j,k] should equal C[i,k].
    
    C_check = np.zeros((2,2))
    for i in range(2):
        for k in range(2):
            for j in range(2):
                C_check[i,k] += T[i,j,k] * A[i,j] * B[j,k]
                
    if not np.allclose(C_check, C):
        # Strassen implementation is wrong. Fall back to schoolbook.
        use_strassen = False
    else:
        use_strassen = True

    # Build the full decomposition
    U_mat = np.zeros((nm, R))
    V_mat = np.zeros((mp, R))
    W_mat = np.zeros((pn, R))
    
    for i_start, j_start, k_start, ni, mj, pk, rank, offset in layout:
        if use_strassen and ni == 2 and mj == 2 and pk == 2:
            U_s, V_s, W_s = get_strassen_factors()
            # Copy U_s to U_mat
            for a in range(2):
                for b in range(2):
                    U_mat[(i_start+a)*m + (j_start+b), offset:offset+7] = U_s[a*2+b, :]
            # Copy V_s to V_mat
            for b in range(2):
                for c in range(2):
                    V_mat[(j_start+b)*p + (k_start+c), offset:offset+7] = V_s[b*2+c, :]
            # Copy W_s to W_mat
            for c in range(2):
                for a in range(2):
                    W_mat[(k_start+c)*n + (i_start+a), offset:offset+7] = W_s[c*2+a, :]
        else:
            # Schoolbook
            idx = 0
            for a in range(ni):
                for b in range(mj):
                    for c in range(pk):
                        col = offset + idx
                        U_mat[(i_start+a)*m + (j_start+b), col] = 1.0
                        V_mat[(j_start+b)*p + (k_start+c), col] = 1.0
                        W_mat[(k_start+c)*n + (i_start+a), col] = 1.0
                        idx += 1
                        
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
