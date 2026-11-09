"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 115  # 115 second deadline (5s margin)
    
    # Strassen's rank-7 decomposition for 2x2x2 matrix multiplication
    # The 7 Strassen products and their contributions to C[i,j]
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M5 + M6
    # C[1,0] = M2 + M4
    # C[1,1] = M3 + M7 - M2 - M4
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For the full 2x4x8 problem:
    # A is 2x4, split into A_left (2x2) and A_right (2x2)
    # B is 4x8, split into B_top (2x8) and B_bottom (2x8)
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # We can apply Strassen to each 2x2 @ 2x8 multiplication.
    # For 2x2 @ 2x8, we have 8 columns, so we do 8 independent 2x2 @ 2x1 multiplications.
    # Each 2x2 @ 2x1 uses rank 7 (Strassen), so total rank = 8 * 7 = 56 per half.
    # Two halves: 56 + 56 = 112? No, that's worse than 64.
    
    # Wait, let me reconsider. The schoolbook rank is n*m*p = 2*4*8 = 64.
    # Strassen on 2x2 gives rank 7 instead of 8, a saving of 1 per 2x2 block.
    
    # Better approach: Think of the 2x4x8 as having structure.
    # A[i, k] where i in {0,1}, k in {0,1,2,3}
    # B[k, j] where k in {0,1,2,3}, j in {0,...,7}
    # C[i, j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    # Split k into two groups: k in {0,1} and k in {2,3}
    # C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    #        = (A_left @ B_top)[i,j] + (A_right @ B_bottom)[i,j]
    
    # For A_left @ B_top: A_left is 2x2, B_top is 2x8
    # This is 2x2 @ 2x8. We can view this as 8 independent 2x2 @ 2x1 problems.
    # Each 2x2 @ 2x1 has tensor T[i,k,j] where i in {0,1}, k in {0,1}, j in {0}
    # That's just the 2x2 @ 2x1 tensor, which has 4 entries all equal to 1.
    # The tensor rank of this is 4 (schoolbook) or potentially 3.
    
    # Actually, let me think about this differently.
    # The 2x2 @ 2x8 multiplication: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # The tensor has entries T[i,k,j] = 1 for all i in {0,1}, k in {0,1}, j in {0,...,7}
    # This is a 2x2x8 tensor.
    
    # We can use Strassen's decomposition for the 2x2 part.
    # Strassen: For matrices X (2x2), Y (2x2), Z = X@Y
    # Z[0,0] = M1 + M4 - M5 - M7
    # Z[0,1] = M5 + M6
    # Z[1,0] = M2 + M4
    # Z[1,1] = M3 + M7 - M2 - M4
    # where M1..M7 are as defined above.
    
    # For 2x2 @ 2x8: Let X = A_left (2x2), Y = B_top (2x8)
    # We can think of Y as two 2x4 matrices, or as 8 columns of 2x1.
    # Actually, let's use the bilinear structure.
    
    # The key insight: 2x2 @ 2x8 can be decomposed using 7 Strassen products
    # where each product involves a 2x1 vector from A and a 2x8 matrix from B,
    # or a 1x2 row and a 2x8 matrix, etc.
    
    # More precisely, for the 2x2 @ 2x8 case:
    # Let A = [[a00, a01], [a10, a11]] (2x2)
    # Let B = [[b0j], [b1j]] for each column j (2x8)
    # C[i,j] = a_i0 * b_0j + a_i1 * b_1j
    
    # Using Strassen:
    # M1 = (a00+a11) * (b0j + b1j)  -- but this is per column j
    # Actually, we need to be more careful.
    
    # Let me use a different formulation. For the 2x2 @ 2x8:
    # We have 7 Strassen products, each of which is a product of a linear
    # combination of A entries and a linear combination of B entries.
    # For B being 2x8, each "B entry" is actually a row of B (1x8).
    
    # So:
    # M1_j = (a00+a11) * (B[0,j] + B[1,j])  for each j
    # M2_j = (a10+a11) * B[0,j]
    # M3_j = a00 * (B[1,j] - B[0,j])
    # M4_j = a11 * (B[0,j] - B[1,j])
    # M5_j = (a00+a01) * B[1,j]
    # M6_j = (a10-a00) * (B[0,j] + B[1,j])
    # M7_j = (a01-a11) * (B[1,j] - B[0,j])
    
    # Then:
    # C[0,j] = M1_j + M4_j - M5_j - M7_j
    # C[1,j] = M2_j + M4_j  ... wait, let me recheck Strassen.
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M4
    # C11 = M3 + M7 - M2 - M4
    
    # For 2x2 @ 2x8, we treat B as 2x8, so B00 becomes B[0,:] (1x8), etc.
    # But the indices in Strassen refer to the 2x2 sub-blocks.
    # Let me relabel: A is 2x2 with entries A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x8 with entries B[0,j], B[1,j] for j=0..7
    
    # In Strassen notation for 2x2 @ 2x2:
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    
    # For 2x2 @ 2x8, we extend B to have 8 columns.
    # The Strassen products become:
    # M1 = (A[0,0]+A[1,1]) * (B[0,:] + B[1,:])  -- 1x8 result
    # M2 = (A[1,0]+A[1,1]) * B[0,:]              -- 1x8 result
    # M3 = A[0,0] * (B[1,:] - B[0,:])            -- 1x8 result
    # M4 = A[1,1] * (B[0,:] - B[1,:])            -- 1x8 result
    # M5 = (A[0,0]+A[0,1]) * B[1,:]              -- 1x8 result
    # M6 = (A[1,0]-A[0,0]) * (B[0,:] + B[1,:])   -- 1x8 result
    # M7 = (A[0,1]-A[1,1]) * (B[1,:] - B[0,:])   -- 1x8 result
    
    # Then:
    # C[0,j] = M1[j] + M4[j] - M5[j] - M7[j]
    # C[1,j] = M2[j] + M4[j]  ... hmm, this doesn't look right.
    
    # Let me re-derive. For the 2x2 @ 2x2 case with B being 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen gives:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M5 + M6
    # C[1,0] = M2 + M4
    # C[1,1] = M3 + M7 - M2 - M4
    
    # For 2x2 @ 2x8, we have 8 independent columns. For each column j:
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # This is like a 2x2 @ 2x1 for each column. The Strassen decomposition
    # for 2x2 @ 2x1 would be:
    # For a single column, B_col = [B[0,j], B[1,j]]^T
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # The tensor for 2x2 @ 2x1 has 4 entries (i,k,j) with j fixed.
    # T[i,k,j] = 1 for all i,k. This is just the 2x2 tensor.
    # The tensor rank of the 2x2 @ 2x1 is 4 (it's just the sum of 4 rank-1 tensors).
    # Can we do it in rank 3? The border rank is 3, but tensor rank might be 4.
    
    # Actually, for the full 2x2 @ 2x8, the tensor is 2x2x8.
    # We can decompose it as: for each of the 8 columns, we have a 2x2 @ 2x1 tensor.
    # If each 2x2 @ 2x1 has rank 4, total rank = 8*4 = 32.
    # If we can share some factors across columns, we might do better.
    
    # Let me try a direct approach: construct the rank-56 decomposition
    # by applying Strassen to the 2x2 block structure.
    
    # The 2x4x8 multiplication:
    # A is 2x4: A[i, k] for i in {0,1}, k in {0,1,2,3}
    # B is 4x8: B[k, j] for k in {0,1,2,3}, j in {0,...,7}
    # C is 2x8: C[i, j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    # Split: A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x8), B_bottom = B[2:4, :] (2x8)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top (2x2 @ 2x8):
    # Using Strassen, we get 7 products, each producing a 1x8 vector.
    # These 7 products are:
    # P1 = (A[0,0]+A[1,1]) * (B[0,:] + B[1,:])
    # P2 = (A[1,0]+A[1,1]) * B[0,:]
    # P3 = A[0,0] * (B[1,:] - B[0,:])
    # P4 = A[1,1] * (B[0,:] - B[1,:])
    # P5 = (A[0,0]+A[0,1]) * B[1,:]
    # P6 = (A[1,0]-A[0,0]) * (B[0,:] + B[1,:])
    # P7 = (A[0,1]-A[1,1]) * (B[1,:] - B[0,:])
    
    # Then:
    # C_top[0,:] = P1 + P4 - P5 - P7
    # C_top[1,:] = P2 + P4  ... wait, I need to be more careful.
    
    # Let me verify with the 2x2 @ 2x2 case first.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-e)
    # M4 = d(f-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C[0,0] = M1 + M4 - M5 - M7
    #         = (a+d)(e+h) + d(f-h) - (a+b)h - (b-d)(g+h)
    #         = ae + ah + de + dh + df - dh - ah - bh - bg - bh + dg + dh
    #         = ae + de + df - bh - bg + dg
    # Hmm, this doesn't simplify to ae+bg. Let me recheck.
    
    # Actually, I think I have the Strassen formula slightly wrong. Let me use the standard one.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M4
    # C11 = M3 + M7 - M2 - M4
    
    # Let me verify C00:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M4 = A11(B01-B11) = A11*B01 - A11*B11
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    # M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # C00 = M1 + M4 - M5 - M7
    #      = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B11
    #        - A00*B11 - A01*B11 - A01*B10 - A01*B11 + A11*B10 + A11*B11
    #      = A00*B00 + A11*B00 + A11*B01 - A01*B11 - A01*B10 + A11*B10
    # This is not equal to A00*B00 + A01*B10. So my Strassen formula is wrong.
    
    # Let me look up the correct Strassen formula.
    # The correct Strassen for 2x2:
    # M1 = (A11 - A01) * (B01 + B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = (A11 - A10) * (B00 + B10)
    # M5 = (A00 - A11) * (B01 + B11)
    # M6 = (A01 + A11) * (B10 - B00)
    # M7 = (A00 + A10) * (B01 - B10)
    # C00 = M5 + M4 + M6 - M7
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 + M5 + M7 - M3
    
    # Let me verify C00 = A00*B00 + A01*B10:
    # M5 = (A00-A11)(B01+B11) = A00*B01 + A00*B11 - A11*B01 - A11*B11
    # M4 = (A11-A10)(B00+B10) = A11*B00 + A11*B10 - A10*B00 - A10*B10
    # M6 = (A01+A11)(B10-B00) = A01*B10 - A01*B00 + A11*B10 - A11*B00
    # M7 = (A00+A10)(B01-B10) = A00*B01 - A00*B10 + A10*B01 - A10*B10
    # C00 = M5 + M4 + M6 - M7
    #      = A00*B01 + A00*B11 - A11*B01 - A11*B11
    #        + A11*B00 + A11*B10 - A10*B00 - A10*B10
    #        + A01*B10 - A01*B00 + A11*B10 - A11*B00
    #        - A00*B01 + A00*B10 - A10*B01 + A10*B10
    #      = (A00*B01 - A00*B01) + A00*B11 - A11*B01 - A11*B11
    #        + A11*B00 - A11*B00 + A11*B10 + A11*B10
    #        - A10*B00 - A10*B10 + A01*B10 - A01*B00
    #        + A00*B10 - A10*B01 + A10*B10
    #      = A00*B11 - A11*B01 - A11*B11 + 2*A11*B10 - A10*B00
    #        + A01*B10 - A01*B00 + A00*B10 - A10*B01
    # This is getting messy. I think I'm making errors in the algebra.
    
    # Let me just use a known correct Strassen decomposition and verify numerically.
    
    def strassen_2x2():
        """Return the Strassen rank-7 decomposition for 2x2x2.
        Returns U, V, W matrices of shape (4,7) each.
        U[i*2+k, r] corresponds to A[i,k]
        V[k*2+j, r] corresponds to B[k,j]
        W[j*2+i, r] corresponds to C[i,j] (using W[j*n+i] convention with n=2)
        """
        # Strassen's 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M5 + M6
        # C10 = M2 + M4
        # C11 = M3 + M7 - M2 - M4
        
        # Let's verify this is correct by expanding.
        # Actually, let me just use a different well-known formulation.
        
        # I'll use the formulation from Wikipedia:
        # M1 = (A11 − A01) × (B01 + B11)
        # M2 = (A00 + A01) × B11
        # M3 = (A10 + A11) × B00
        # M4 = (A11 − A10) × (B00 + B10)
        # M5 = (A00 − A11) × (B01 + B11)
        # M6 = (A01 + A11) × (B10 − B00)
        # M7 = (A00 + A10) × (B01 − B10)
        # C00 = M5 + M4 + M6 − M7
        # C01 = M1 + M2
        # C10 = M3 + M4
        # C11 = M1 + M5 + M7 − M3
        
        # Let me verify C00 = A00*B00 + A01*B10:
        # M5 = (A00-A11)(B01+B11)
        # M4 = (A11-A10)(B00+B10)
        # M6 = (A01+A11)(B10-B00)
        # M7 = (A00+A10)(B01-B10)
        
        # I'll just test numerically.
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        # W[j*n+i, r] with n=2: W[0,0]=C[0,0], W[1,0]=C[0,1], W[2,0]=C[1,0], W[3,0]=C[1,1]
        # So W[j*2+i]: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        
        # M1 = (A11-A01)(B01+B11)
        # A part: A11 - A01 = A[1,1] - A[0,1]
        # B part: B01 + B11 = B[0,1] + B[1,1]
        U[3, 0] = 1   # A[1,1]
        U[1, 0] = -1  # A[0,1]
        V[1, 0] = 1   # B[0,1]
        V[3, 0] = 1   # B[1,1]
        
        # M2 = (A00+A01)B11
        U[0, 1] = 1   # A[0,0]
        U[1, 1] = 1   # A[0,1]
        V[3, 1] = 1   # B[1,1]
        
        # M3 = (A10+A11)B00
        U[2, 2] = 1   # A[1,0]
        U[3, 2] = 1   # A[1,1]
        V[0, 2] = 1   # B[0,0]
        
        # M4 = (A11-A10)(B00+B10)
        U[3, 3] = 1   # A[1,1]
        U[2, 3] = -1  # A[1,0]
        V[0, 3] = 1   # B[0,0]
        V[2, 3] = 1   # B[1,0]
        
        # M5 = (A00-A11)(B01+B11)
        U[0, 4] = 1   # A[0,0]
        U[3, 4] = -1  # A[1,1]
        V[1, 4] = 1   # B[0,1]
        V[3, 4] = 1   # B[1,1]
        
        # M6 = (A01+A11)(B10-B00)
        U[1, 5] = 1   # A[0,1]
        U[3, 5] = 1   # A[1,1]
        V[2, 5] = 1   # B[1,0]
        V[0, 5] = -1  # B[0,0]
        
        # M7 = (A00+A10)(B01-B10)
        U[0, 6] = 1   # A[0,0]
        U[2, 6] = 1   # A[1,0]
        V[1, 6] = 1   # B[0,1]
        V[2, 6] = -1  # B[1,0]
        
        # C00 = M5 + M4 + M6 - M7
        W[0, 4] = 1   # M5
        W[0, 3] = 1   # M4
        W[0, 5] = 1   # M6
        W[0, 6] = -1  # M7
        
        # C01 = M1 + M2
        W[1, 0] = 1   # M1
        W[1, 1] = 1   # M2
        
        # C10 = M3 + M4
        W[2, 2] = 1   # M3
        W[2, 3] = 1   # M4
        
        # C11 = M1 + M5 + M7 - M3
        W[3, 0] = 1   # M1
        W[3, 4] = 1   # M5
        W[3, 6] = 1   # M7
        W[3, 2] = -1  # M3
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_2x2()
    
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication"""
        rank_val = len(U_mat[0])
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    coeff = np.sum(U_arr[i*m+k, :] * V_arr[k*p+j, :] * W_arr[j*n+i, :])
                    if abs(coeff - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_decomposition(U_s.tolist(), V_s.tolist(), W_s.tolist(), 2, 2, 2):
        # Fallback to schoolbook if Strassen is wrong
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now construct the 2x4x8 decomposition using Strassen.
    # A is 2x4: A[i, k] for i in {0,1}, k in {0,1,2,3}
    # B is 4x8: B[k, j] for k in {0,1,2,3}, j in {0,...,7}
    # C is 2x8: C[i, j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    
    # Split into two 2x2 @ 2x8 multiplications:
    # C = A_left @ B_top + A_right @ B_bottom
    # where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    
    # For A_left @ B_top (2x2 @ 2x8):
    # Use Strassen with 7 products, each producing a 1x8 vector.
    # The 7 products are linear combinations of A_left entries times linear combinations of B_top rows.
    
    # For A_right @ B_bottom (2x2 @ 2x8):
    # Similarly, 7 products.
    
    # Total rank = 7 + 7 = 14? No, each product is a scalar times a vector,
    # so we need to represent each product as a rank-1 term in the tensor decomposition.
    
    # Actually, each Strassen product M_r = (linear combo of A) * (linear combo of B)
    # where the "linear combo of B" is a 1x8 vector.
    # In the tensor decomposition, each rank-1 term has:
    # U[i*m+k, r] = coefficient of A[i,k] in the linear combo
    # V[k*p+j, r] = coefficient of B[k,j] in the linear combo
    # W[j*n+i, r] = coefficient in the output
    
    # For A_left @ B_top:
    # A_left[i, k] for i in {0,1}, k in {0,1}
    # B_top[k, j] for k in {0,1}, j in {0,...,7}
    # The Strassen products use A_left and B_top.
    
    # Let me index the full A and B:
    # A[i, k] for i in {0,1}, k in {0,1,2,3}
    # A_left[i, k] = A[i, k] for k in {0,1}
    # A_right[i, k] = A[i, k] for k in {2,3}
    
    # B[k, j] for k in {0,1,2,3}, j in {0,...,7}
    # B_top[k, j] = B[k, j] for k in {0,1}
    # B_bottom[k, j] = B[k, j] for k in {2,3}
    
    # For A_left @ B_top, the Strassen products are:
    # M1 = (A[1,1]-A[0,1]) * (B[0,1]+B[1,1])  -- but B is 2x8, so B[0,1] means B[0, :]? No.
    
    # Wait, I need to be more careful. In the 2x2 @ 2x8 case:
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 2x8: B[0,0], B[0,1], ..., B[0,7], B[1,0], B[1,1], ..., B[1,7]
    # C is 2x8: C[0,0], C[0,1], ..., C[0,7], C[1,0], C[1,1], ..., C[1,7]
    
    # The Strassen decomposition for 2x2 @ 2x2 generalizes to 2x2 @ 2x8
    # by treating each column of B independently.
    
    # For each column j of B:
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # The Strassen products for column j:
    # M1_j = (A[1,1]-A[0,1]) * (B[0,j]+B[1,j])
    # M2_j = (A[0,0]+A[0,1]) * B[1,j]
    # M3_j = (A[1,0]+A[1,1]) * B[0,j]
    # M4_j = (A[1,1]-A[1,0]) * (B[0,j]+B[1,j])
    # M5_j = (A[0,0]-A[1,1]) * (B[0,j]+B[1,j])
    # M6_j = (A[0,1]+A[1,1]) * (B[1,j]-B[0,j])
    # M7_j = (A[0,0]+A[1,0]) * (B[0,j]-B[1,j])
    
    # Then:
    # C[0,j] = M5_j + M4_j + M6_j - M7_j
    # C[1,j] = M1_j + M5_j + M7_j - M3_j
    
    # Wait, this doesn't look right either. Let me just use the verified Strassen
    # decomposition and extend it to 2x8 by replicating for each column.
    
    # Actually, the cleanest way is to note that 2x2 @ 2x8 is just 8 independent
    # 2x2 @ 2x1 multiplications, one for each column. Each 2x2 @ 2x1 can be done
    # with rank 7 (Strassen), but that would give rank 8*7 = 56 for the 2x2 @ 2x8.
    # But we can share the A factors across columns!
    
    # The key insight: In the tensor decomposition, each rank-1 term has a fixed
    # pattern in U (for A) and V (for B) and W (for C). If we use Strassen,
    # the U factors are the same for all 8 columns, and the V factors differ per column.
    
    # So for A_left @ B_top:
    # We have 7 Strassen "products", but each product involves a linear combination
    # of A entries (same for all columns) and a linear combination of B entries
    # (different for each column).
    
    # In the tensor decomposition, each rank-1 term corresponds to one specific
    # (i, k, j) combination. So for the 2x2 @ 2x8, we have 2*2*8 = 32 entries
    # in the tensor, and we want to decompose it into rank-1 terms.
    
    # Using Strassen, we can group the 32 entries into 7 groups, where each group
    # has 8 entries (one for each column j). But each rank-1 term only covers one
    # (i, k, j) combination, so we still need 8 terms per Strassen product, giving
    # 7*8 = 56 terms for the 2x2 @ 2x8.
    
    # Hmm, that's not right either. Let me think about this more carefully.
    
    # In the tensor decomposition T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # (with appropriate indexing), each rank-1 term (r) contributes to all (i,k,j)
    # where U[i,k,r], V[k,j,r], and W[j,i,r] are all non-zero.
    
    # For Strassen's 2x2 @ 2x2, each of the 7 rank-1 terms has:
    # U with 2 non-zero entries (in the A part)
    # V with 2 non-zero entries (in the B part)
    # W with 1 non-zero entry (in the C part)
    
    # For 2x2 @ 2x8, if we extend Strassen, each rank-1 term would have:
    # U with 2 non-zero entries (same as 2x2 case)
    # V with 2*8 = 16 non-zero entries? No, V[k,j,r] for k in {0,1}, j in {0,...,7}
    # If the Strassen product uses B[0,j] and B[1,j] for all j, then V has 16 non-zero entries.
    # W with 2*8 = 16 non-zero entries? W[j,i,r] for j in {0,...,7}, i in {0,1}
    
    # This is getting complicated. Let me just directly construct the decomposition.
    
    # For the full 2x4x8 problem, let's use the following approach:
    # Split A into A_left (2x2) and A_right (2x2)
    # Split B into B_top (2x8) and B_bottom (2x8)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top, use Strassen with 7 products.
    # Each product M_r = (linear combo of A_left) * (linear combo of B_top)
    # The linear combo of B_top is a 1x8 vector.
    # In the tensor decomposition, this means:
    # U[i*m+k, r] = coefficient of A[i,k] in the linear combo (for k in {0,1})
    # V[k*p+j, r] = coefficient of B[k,j] in the linear combo (for k in {0,1}, j in {0,...,7})
    # W[j*n+i, r] = coefficient in the output (for j in {0,...,7}, i in {0,1})
    
    # But wait, in the tensor decomposition, each rank-1 term is a single
    # (i,k,j) triple. So we can't have a rank-1 term that covers multiple j values.
    
    # I think the issue is that Strassen's decomposition is for the bilinear map,
    # not for the tensor in the specific indexing used here.
    
    # Let me try a different approach: just use the schoolbook decomposition
    # and try to reduce the rank by finding linear dependencies.
    
    # Actually, let me just try to construct a rank-56 decomposition by
    # applying Strassen to the 2x2 block structure in a way that's compatible
    # with the tensor indexing.
    
    # The tensor T[i,k,j] = 1 for all i in {0,1}, k in {0,1,2,3}, j in {0,...,7}
    # We want T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where m=4, p=8, n=2
    # So U has shape (8, r), V has shape (32, r), W has shape (16, r)
    # U[i*4+k, r], V[k*8+j, r], W[j*2+i, r]
    
    # Let me try to construct this using Strassen for the 2x2 @ 2x2 core.
    # The 2x4x8 can be seen as 4 independent 2x2x8 problems? No.
    # It's 2x4 @ 4x8 = 2x8.
    
    # Let me just try to find a rank-56 decomposition by solving a system of equations.
    # We have 64 equations (one for each (i,k,j)) and 56*r unknowns.
    # For r=56, we have 56*56 = 3136 unknowns, which is much more than 64 equations.
    # So the system is very underdetermined, and a solution should exist.
    
    # Let me try to construct it by setting most entries to 0 and solving for the rest.
    
    # Actually, let me just use the Strassen decomposition directly.
    # For the 2x2 @ 2x2 case, Strassen gives rank 7.
    # For the 2x4 @ 4x8 case, we can think of it as:
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1 @ B1 + A2 @ B2
    
    # Each A_i @ B_i is a 2x2 @ 2x8 multiplication.
    # The 2x2 @ 2x8 can be decomposed using Strassen with rank 7*8 = 56? No.
    
    # Let me just try to construct a valid decomposition with rank 56.
    # I'll use the following strategy:
    # For each of the 8 columns j of B, we have a 2x4 @ 4x1 multiplication.
    # C[:,j] = A @ B[:,j]
    # This is a 2x4 @ 4x1 = 2x1 multiplication.
    # The tensor for 2x4 @ 4x1 has 2*4*1 = 8 entries, all equal to 1.
    # The tensor rank of this is 8 (schoolbook).
    # Can we do it in rank < 8? The tensor is T[i,k,j] with j fixed, so it's
    # just a 2x4 matrix of 1s. The tensor rank of a 2x4 matrix of 1s is 4
    # (since it's rank 4 as a matrix, and the tensor rank is at least the matrix rank).
    # Actually, the tensor T[i,k] = 1 for all i,k has tensor rank 4 (it's the
    # sum of 4 rank-1 tensors: e_i * e_k for each (i,k) pair).
    # Wait, no. The tensor T[i,k] = 1 can be written as sum_{r=1}^{4} U[i,r] * V[k,r]
    # where U is 2x4 and V is 4x4. For example, U = [[1,1,1,1],[1,1,1,1]] and
    # V = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]. Then T[i,k] = sum_r U[i,r]*V[k,r] = 4.
    # That's not right. We need T[i,k] = 1, not 4.
    # So U = [[1/4, 1/4, 1/4, 1/4], [1/4, 1/4, 1/4, 1/4]] and V = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]].
    # Then T[i,k] = sum_r (1/4)*(1) = 4*(1/4) = 1. Yes!
    # So the tensor rank of the 2x4 matrix of 1s is 4.
    
    # But in our problem, the tensor is 3D: T[i,k,j] for i in {0,1}, k in {0,1,2,3}, j in {0,...,7}.
    # And T[i,k,j] = 1 for all (i,k,j).
    # This is a 2x4x8 tensor of all 1s.
    # The tensor rank of an all-1s tensor of size a x b x c is max(a,b,c) if we can
    # factor it as sum_r U[i,r] * V[k,r] * W[j,r].
    # For 2x4x8, the tensor rank is at least 8 (the largest dimension).
    # And it can be achieved: U[i,r] = 1/2 for all i,r (r=0..7), V[k,r] = 1/4 for all k,r,
    # W[j,r] = 1 for all j,r. Then T[i,k,j] = sum_r (1/2)*(1/4)*(1) = 8*(1/8) = 1.
    # So the tensor rank of the all-1s 2x4x8 tensor is 8!
    
    # But wait, that's not the matrix multiplication tensor. The matrix multiplication
    # tensor is T[i,k,j] = 1 if and only if the (i,k) entry of A and the (k,j) entry of B
    # contribute to the (i,j) entry of C. In standard matrix multiplication,
    # C[i,j] = sum_k A[i,k] * B[k,j], so the tensor is T[i,k,j] = 1 for all valid (i,k,j).
    # So yes, the matrix multiplication tensor for 2x4 @ 4x8 is indeed the all-1s tensor
    # of size 2x4x8.
    
    # And the tensor rank of the all-1s 2x4x8 tensor is 8!
    
    # So we can achieve rank 8! Let me construct this.
    
    # U[i*m+k, r] = U[i*4+k, r] for i in {0,1}, k in {0,1,2,3}, r in {0,...,7}
    # V[k*p+j, r] = V[k*8+j, r] for k in {0,1,2,3}, j in {0,...,7}, r in {0,...,7}
    # W[j*n+i, r] = W[j*2+i, r] for j in {0,...,7}, i in {0,1}, r in {0,...,7}
    
    # We want: sum_{r=0}^{7} U[i*4+k, r] * V[k*8+j, r] * W[j*2+i, r] = 1 for all i,k,j
    
    # Let U[i*4+k, r] = 1/2 for all i,k,r
    # V[k*8+j, r] = 1/4 for all k,j,r
    # W[j*2+i, r] = 1 for all j,i,r
    # Then sum_r (1/2)*(1/4)*(1) = 8*(1/8) = 1. 
    
    # So rank 8 works! Let me construct this.
    
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for k in range(m):
                U[i * m + k][r] = 1.0 / 2
        for k in range(m):
            for j in range(p):
                V[k * p + j][r] = 1.0 / 4
        for j in range(p):
            for i in range(n):
                W[j * n + i][r] = 1.0
    
    # Verify
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
