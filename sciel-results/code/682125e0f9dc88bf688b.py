import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Use the tensor identity M_{4,4,6} ≅ M_{2,2,3} ⊗ M_{2,2,2}
    # Rank = rank(M_{2,2,3}) * rank(M_{2,2,2}) = 6 * 7 = 42
    # 
    # Actually, let me think more carefully. We need to multiply a 4x4 matrix by a 4x6 matrix.
    # We can view this as:
    # - Split A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    # - Split B (4x6) into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x3
    #
    # Then C = A*B is 4x6, split into 2x3 blocks: C = [[C00, C01, C02], [C10, C11, C12]]
    # where Cij = A00*B0j + A01*B1j (for i=0) and Cij = A10*B0j + A11*B1j (for i=1)
    #
    # Each Cij is 2x3. We can compute each 2x3 product using the rank-6 decomposition of M_{2,2,3}.
    # But that would give us 4 * 6 = 24 multiplications of scalars... no wait.
    #
    # Actually, let me use a different approach. The standard way to get rank 42:
    # 
    # View the 4x4 x 4x6 multiplication as a Kronecker product structure.
    # Let's use: M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # 
    # The tensor for M_{2,2,3} has rank 6. Let's denote its decomposition as:
    # T_{2,2,3} = sum_{r=0}^{5} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{2*2}=R^4, v_r is in R^{2*3}=R^6, w_r is in R^{2*3}=R^6
    #
    # The tensor for M_{2,2,2} has rank 7 (Strassen). Let's denote:
    # T_{2,2,2} = sum_{s=0}^{6} u'_s ⊗ v'_s ⊗ w'_s
    # where u'_s, v'_s are in R^4, w'_s is in R^4
    #
    # The Kronecker product tensor T_{4,4,6} = T_{2,2,3} ⊗ T_{2,2,2} has rank 6*7=42.
    # 
    # For the Kronecker product:
    # U_total[:, (r,s)] = u_r ⊗ u'_s  (length 16)
    # V_total[:, (r,s)] = v_r ⊗ v'_s  (length 24)
    # W_total[:, (r,s)] = w_r ⊗ w'_s  (length 24)
    #
    # Let me first establish the rank-6 decomposition of M_{2,2,3}.
    # For multiplying a 2x2 matrix X by a 2x3 matrix Y to get a 2x3 matrix Z:
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Z[0,2] = X[0,0]*Y[0,2] + X[0,1]*Y[1,2]
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    # Z[1,2] = X[1,0]*Y[0,2] + X[1,1]*Y[1,2]
    #
    # This is 6 scalar multiplications (schoolbook). So rank(M_{2,2,3}) <= 6.
    # Actually, it's known that rank(M_{2,2,3}) = 6 (it's minimal).
    #
    # So the 6 terms for M_{2,2,3} schoolbook:
    # Term 0: X[0,0]*Y[0,0] -> contributes to Z[0,0]
    # Term 1: X[0,1]*Y[1,0] -> contributes to Z[0,0]
    # Term 2: X[0,0]*Y[0,1] -> contributes to Z[0,1]
    # Term 3: X[0,1]*Y[1,1] -> contributes to Z[0,1]
    # Term 4: X[0,0]*Y[0,2] -> contributes to Z[0,2]
    # Term 5: X[0,1]*Y[1,2] -> contributes to Z[0,2]
    # Wait, that's only 6 but we need to cover all 6 entries of Z.
    # Actually schoolbook for 2x2 * 2x3 needs 2*2*3 = 12 multiplications if done naively.
    # But we can do it with 6:
    # Z[0,:] = X[0,0]*Y[0,:] + X[0,1]*Y[1,:]  (this is 2*3 = 6 mults)
    # Z[1,:] = X[1,0]*Y[0,:] + X[1,1]*Y[1,:]  (this is 2*3 = 6 mults)
    # Total 12. Hmm.
    
    # Let me reconsider. The rank of M_{2,2,3} is actually 6? Let me check.
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x2 * 2x3, we can think of it as two 2x2 * 2x1 problems plus... no.
    # Actually, 2x2 * 2x3: the output is 2x3 = 6 entries, each is a sum of 2 products.
    # The minimum rank... I believe it's 6.
    # 
    # Wait, actually I think the rank of 2x2 * 2x3 matrix multiplication is 6.
    # Here's a rank-6 construction:
    # p1 = X[0,0]*Y[0,0]
    # p2 = X[0,1]*Y[1,0]
    # p3 = X[0,0]*Y[0,1]
    # p4 = X[0,1]*Y[1,1]
    # p5 = X[0,0]*Y[0,2]
    # p6 = X[0,1]*Y[1,2]
    # But this doesn't cover Z[1,:]. We need:
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    # Z[1,2] = X[1,0]*Y[0,2] + X[1,1]*Y[1,2]
    # So we need 12 terms total for schoolbook. The rank is at least 6 (since output has 6 entries)
    # and at most 12.
    
    # Let me just use the schoolbook construction with 12 terms for M_{2,2,3},
    # then the Kronecker product with M_{2,2,2} (rank 7) gives rank 12*7 = 84. That's worse.
    
    # Alternative: Use the fact that M_{4,4,6} can be decomposed differently.
    # Let me just use the nested Strassen approach but more carefully.
    
    # Actually, let me try a direct approach. The current program has rank 77.
    # Let me try to use a better decomposition.
    
    # For 4x4 * 4x6:
    # One approach: split into 2x2 * 2x3 blocks.
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]] (each 2x3)
    # C = [[C00, C01, C02], [C10, C11, C12]] (each 2x3)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each Cij is a 2x3 matrix computed as sum of two 2x2 * 2x3 products.
    # If we use Strassen for each 2x2 * 2x3 product... but Strassen is for 2x2*2x2.
    
    # Let me just go with the Kronecker product approach using known decompositions.
    # 
    # M_{2,2,2} rank 7 (Strassen)
    # M_{2,2,3} rank 6 (need to verify)
    
    # Actually, I recall that the rank of 2x2 * 2x3 is indeed 6. Here's the construction:
    # We can write the 2x3 output as:
    # Z = X * Y where X is 2x2, Y is 2x3
    # Let's define:
    # p1 = X[0,0] * Y[0,0]
    # p2 = X[0,0] * Y[0,1]  
    # p3 = X[0,0] * Y[0,2]
    # p4 = X[0,1] * Y[1,0]
    # p5 = X[0,1] * Y[1,1]
    # p6 = X[0,1] * Y[1,2]
    # But this only gives Z[0,:]. For Z[1,:]:
    # p7 = X[1,0] * Y[0,0]
    # p8 = X[1,0] * Y[0,1]
    # p9 = X[1,0] * Y[0,2]
    # p10 = X[1,1] * Y[1,0]
    # p11 = X[1,1] * Y[1,1]
    # p12 = X[1,1] * Y[1,2]
    # That's 12 terms.
    
    # I think the rank of M_{2,2,3} is actually 6 is incorrect. Let me just use the 
    # nested approach from the current program but verify it works, and try to optimize.
    
    # Let me stick with the current approach (rank 77) and make sure it's correct,
    # then see if I can find a better decomposition.
    
    # Actually, let me try a completely different approach: use the fact that
    # 4x4 * 4x6 can be viewed as computing 4x6 output where each row is a linear combination.
    # 
    # Let me just implement the rank-42 Kronecker product construction properly.
    # 
    # Key insight: M_{4,4,6} is NOT exactly M_{2,2,3} ⊗ M_{2,2,2} in a simple way.
    # Let me think about this differently.
    
    # The matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1 if j=k (in the 
    # appropriate indexing). For our case, we're computing C = A*B where A is 4x4, B is 4x6.
    
    # Let me use a block decomposition:
    # A = [[A0, A1], [A2, A3]] where Ai are 2x2
    # B = [[B0, B1], [B2, B3]] where Bi are 2x3
    # C = [[C0, C1], [C2, C3]] where Ci are 2x3
    # C0 = A0*B0 + A1*B2
    # C1 = A0*B1 + A1*B3
    # C2 = A2*B0 + A3*B2
    # C3 = A2*B1 + A3*B3
    
    # For each Ci, we need to compute a sum of two 2x2 * 2x3 products.
    # 
    # Let me use Strassen-like decomposition for 2x2 * 2x3.
    # Actually, I'll just use the schoolbook for 2x2 * 2x3 (12 mults each) and
    # Strassen for the outer 2x2 block structure (7 mults of 2x2*2x3).
    # Total: 7 * 12 = 84. That's worse than 77.
    
    # The current program uses 7 * (7 + 4) = 7 * 11 = 77, where for each 2x2*2x3:
    # - 7 mults for the 2x2*2x2 part (columns 0,1)
    # - 4 mults for the 2x2*2x1 part (column 2)
    # Total 11 per block, 7 blocks = 77.
    
    # Can we do better? What if we use a rank-6 decomposition for 2x2*2x3?
    # If rank(M_{2,2,3}) = 6, then 7 * 6 = 42.
    
    # Let me look up or derive the rank-6 decomposition of 2x2 * 2x3.
    # 
    # For X (2x2) * Y (2x3) = Z (2x3):
    # Z[0,0] = X00*Y00 + X01*Y10
    # Z[0,1] = X00*Y01 + X01*Y11
    # Z[0,2] = X00*Y02 + X01*Y12
    # Z[1,0] = X10*Y00 + X11*Y10
    # Z[1,1] = X10*Y01 + X11*Y11
    # Z[1,2] = X10*Y02 + X11*Y12
    #
    # A rank-6 decomposition would have 6 terms p1..p6.
    # Let me try:
    # p1 = X00 * (Y00 + Y01 + Y02)
    # p2 = X01 * (Y10 + Y11 + Y12)
    # p3 = X10 * (Y00 + Y01 + Y02)
    # p4 = X11 * (Y10 + Y11 + Y12)
    # p5 = (X00 + X01) * (Y10 + Y11 + Y12)  -- no this is getting complicated
    
    # I think the rank of 2x2*2x3 is actually 6. Here's a known decomposition:
    # p1 = X[0,0]*Y[0,0]
    # p2 = X[0,0]*Y[0,1]
    # p3 = X[0,0]*Y[0,2]
    # p4 = X[0,1]*Y[1,0]
    # p5 = X[0,1]*Y[1,1]
    # p6 = X[0,1]*Y[1,2]
    # p7 = X[1,0]*Y[0,0]
    # p8 = X[1,0]*Y[0,1]
    # p9 = X[1,0]*Y[0,2]
    # p10 = X[1,1]*Y[1,0]
    # p11 = X[1,1]*Y[1,1]
    # p12 = X[1,1]*Y[1,2]
    # That's 12. I don't think 6 is achievable with this structure.
    
    # Let me just keep the current rank-77 approach and make sure it's correct.
    # The current program should work. Let me verify and clean it up.
    
    n, m, p = DIMENSIONS
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 * 2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
