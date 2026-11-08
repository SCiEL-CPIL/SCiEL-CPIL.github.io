import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # M_{4,4,6} = M_{2,2,2} (outer blocks) ⊗ M_{2,2,3} (inner)
    # Outer: schoolbook 2x2 * 2x2, rank 4
    # Inner: schoolbook 2x2 * 2x3, rank 6 (2 per column * 3 columns)
    # Total rank: 4 * 6 = 24
    
    R = 24
    U = np.zeros((16, R))  # n*m = 4*4 = 16
    V = np.zeros((24, R))  # m*p = 4*6 = 24
    W = np.zeros((24, R))  # p*n = 6*4 = 24
    
    # Outer schoolbook: 4 terms
    # s=0: A00 * B00 -> C00
    # s=1: A01 * B10 -> C01  (wait, need to be careful)
    # For 2x2 * 2x2 schoolbook:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # So 8 terms for full schoolbook. But we want rank 4...
    # 
    # Wait, rank 4 for 2x2*2x2 means we use 4 rank-1 tensors.
    # Schoolbook for 2x2*2x2 uses 4 multiplications? No, it uses 4*4=16 scalar mults
    # but 4 "block" multiplications if we think of it as 2x2 blocks.
    # 
    # Actually, the rank of M_{2,2,2} as a tensor is 7 (Strassen). 
    # Schoolbook uses 4 terms where each term is a 2x2*2x2 multiplication.
    # But in the tensor decomposition sense, each "block multiplication" A_ik * B_kj
    # is itself a tensor of rank 4 (schoolbook for the inner 2x2*2x2).
    # 
    # Hmm, I'm confusing levels. Let me think more carefully.
    # 
    # The tensor M_{4,4,6} has entries T[i,k,j] = 1 if the (i,k) entry of A and (k,j) entry of B
    # contribute to C[i,j].
    # 
    # Using Kronecker: index i = (i1, i0) where i1 in {0,1}, i0 in {0,1,2,3}... no.
    # Let me use: i = (i1, i0) with i1 in {0,1}, i0 in {0,1} for the 4x4 matrix (2x2 blocks of 2x2).
    # k = (k1, k0) with k1 in {0,1}, k0 in {0,1} for the 4x4 matrix.
    # j = (j1, j0) with j1 in {0,1}, j0 in {0,1,2} for the 4x6 matrix (2x2 blocks of 2x3).
    # 
    # T[i1,i0, k1,k0, j1,j0] = T_outer[i1,k1,j1] * T_inner[i0,k0,j0]
    # where T_outer is M_{2,2,2} and T_inner is M_{2,2,3}.
    # 
    # A rank-r decomposition of M_{2,2,2} gives factors U_o, V_o, W_o of size 4xr, 4xr, 4xr.
    # A rank-s decomposition of M_{2,2,3} gives factors U_i, V_i, W_i of size 4xs, 6xs, 6xs.
    # 
    # The Kronecker product decomposition has rank r*s with:
    # U[(i1*2+i0)*4 + (k1*2+k0), t] = U_o[i1*2+k1, t1] * U_i[i0*2+k0, t2]
    #   where t = t1*s + t2
    # V[(k1*2+k0)*6 + (j1*3+j0), t] = V_o[k1*2+j1, t1] * V_i[k0*2+j0, t2]
    # W[(j1*3+j0)*4 + (i1*2+i0), t] = W_o[j1*2+i1, t1] * W_i[j0*2+i0, t2]
    # 
    # For M_{2,2,2} with rank 4 (schoolbook):
    # t1 = 0: A00*B00 -> C00
    # t1 = 1: A01*B10 -> C01
    # t1 = 2: A10*B00 -> C10
    # t1 = 3: A11*B10 -> C11
    # Wait, that's not right. Schoolbook for 2x2*2x2:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # 
    # The 4 "block" terms are:
    # t1=0: A00 * B00 (contributes to C00, C10)
    # t1=1: A01 * B10 (contributes to C00, C10)  
    # t1=2: A00 * B01 (contributes to C01, C11)
    # t1=3: A01 * B11 (contributes to C01, C11)
    # Hmm, that's not quite right either. Let me think of it as:
    # The 4 terms correspond to (i1,k1,j1) combinations:
    # t1=0: (0,0,0) -> A00*B00 contributes to C00
    # t1=1: (0,1,0) -> A01*B10 contributes to C00
    # t1=2: (0,0,1) -> A00*B01 contributes to C01
    # t1=3: (0,1,1) -> A01*B11 contributes to C01
    # t1=4: (1,0,0) -> A10*B00 contributes to C10
    # t1=5: (1,1,0) -> A11*B10 contributes to C10
    # t1=6: (1,0,1) -> A10*B01 contributes to C11
    # t1=7: (1,1,1) -> A11*B11 contributes to C11
    # That's 8 terms for schoolbook, not 4.
    #
    # I think I was wrong. The rank of M_{2,2,2} is 7, not 4.
    # Schoolbook uses 8 rank-1 terms (one per A_ik * B_kj product).
    # Strassen uses 7.
    #
    # So for rank 24, I need 6 * 4 = 24, but 4 is not a valid rank for M_{2,2,2}.
    # 
    # Let me reconsider. Maybe the directive means something different.
    # "rank-24 decomposition from the known rank-6 tensor decomposition of 2x2 by 2x3"
    # 
    # What if we use M_{2,2,3} (rank 6) ⊗ M_{2,2,2} (rank 4)? But M_{2,2,2} doesn't have rank 4.
    # 
    # Or maybe: M_{4,4,6} can be seen as M_{2,2,3} ⊗ M_{2,2,2} where:
    # - M_{2,2,3} handles the "inner" 2x2 * 2x3 multiplication (rank 6)
    # - M_{2,2,2} handles the "outer" 2x2 * 2x2 block multiplication
    # 
    # But the outer is M_{2,2,2} which has rank 7. So 6*7=42.
    # 
    # Unless we use a different factorization. What if:
    # M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2} but we use rank 4 for the outer?
    # That's not possible since rank(M_{2,2,2}) >= 7.
    #
    # Let me re-read the directive: "rank-24 decomposition from the known rank-6 tensor decomposition 
    # of 2x2 by 2x3 matrix multiplication (or equivalent rank-12 for 2x2 by 2x2 scaled appropriately)"
    #
    # "equivalent rank-12 for 2x2 by 2x2 scaled appropriately" - maybe they mean using a rank-12 
    # decomposition for some other factor?
    #
    # Or: 6 * 4 = 24 where 4 comes from... hmm.
    #
    # What if the factorization is M_{2,2,3} ⊗ M_{2,2,2} but we interpret it differently?
    # 
    # Actually, wait. Let me reconsider the Kronecker product structure.
    # A is 4x4, B is 4x6, C is 4x6.
    # 
    # What if we split as:
    # A = kron(A_outer, A_inner) where A_outer is 2x2, A_inner is 2x2
    # B = kron(B_outer, B_inner) where B_outer is 2x2, B_inner is 2x3
    # C = kron(C_outer, C_inner) where C_outer is 2x2, C_inner is 2x3
    # 
    # Then C = A*B = kron(A_outer, A_inner) * kron(B_outer, B_inner) = kron(A_outer*B_outer, A_inner*B_inner)
    # So C_outer = A_outer * B_outer (2x2 * 2x2) and C_inner = A_inner * B_inner (2x2 * 2x3).
    # 
    # The tensor is M_{2,2,2} ⊗ M_{2,2,3}.
    # Rank = rank(M_{2,2,2}) * rank(M_{2,2,3}) = 7 * 6 = 42.
    #
    # To get rank 24, we'd need rank(M_{2,2,2}) = 4, which is impossible.
    #
    # I think the directive might be slightly off, or I'm misunderstanding. Let me just implement 
    # the rank-42 version (Strassen outer + schoolbook inner) which is already an improvement 
    # over 77. Actually wait, 42 < 77, so that's better!
    #
    # Let me implement rank 42: Strassen for M_{2,2,2} (rank 7) × schoolbook for M_{2,2,3} (rank 6).
    
    R = 42
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen for M_{2,2,2}: rank 7
    # alpha[s][i1*2+k1] for s=0..6, i1,k1 in {0,1}
    alpha = [
        [1, 0, 0, 1],  # s=0: A00 + A11
        [0, 0, 1, 1],  # s=1: A10 + A11
        [1, 0, 0, 0],  # s=2: A00
        [0, 0, 0, 1],  # s=3: A11
        [1, 1, 0, 0],  # s=4: A00 + A01
        [-1, 0, 1, 0], # s=5: A10 - A00
        [0, 1, 0, -1], # s=6: A11 - A10
    ]
    # beta[s][k1*2+j1] for s=0..6, k1,j1 in {0,1}
    beta = [
        [1, 0, 0, 1],  # s=0: B00 + B11
        [1, 0, 0, 0],  # s=1: B00
        [0, 1, 0, -1], # s=2: B11 - B10
        [-1, 0, 1, 0], # s=3: B10 - B00
        [0, 0, 0, 1],  # s=4: B11
        [1, 1, 0, 0],  # s=5: B00 + B01
        [0, 0, 1, 1],  # s=6: B10 + B11
    ]
    # recon[s] = list of (i1, j1, gamma)
    recon = [
        [(0, 0, 1), (1, 1, 1)],   # C00 += S0, C11 += S0
        [(1, 0, 1), (1, 1, -1)],  # C10 += S1, C11 -= S1
        [(0, 1, 1), (1, 1, 1)],   # C01 += S2, C11 += S2
        [(0, 0, 1), (1, 0, 1)],   # C00 += S3, C10 += S3
        [(0, 0, -1), (0, 1, 1)],  # C00 -= S4, C01 += S4
        [(1, 1, 1)],               # C11 += S5
        [(0, 0, 1)],               # C00 += S6
    ]
    
    # Schoolbook for M_{2,2,3}: rank 6
    # A_inner is 2x2, B_inner is 2x3, C_inner is 2x3
    # C_inner[i0, j0] = A_inner[i0,0]*B_inner[0,j0] + A_inner[i0,1]*B_inner[1,j0]
    # 6 terms: for each (i0, j0) pair, 2 products
    # t2 = 0: A[0,0]*B[0,0] -> C[0,0]
    # t2 = 1: A[0,1]*B[1,0] -> C[0,0]
    # t2 = 2: A[0,0]*B[0,1] -> C[0,1]
    # t2 = 3: A[0,1]*B[1,1] -> C[0,1]
    # t2 = 4: A[1,0]*B[0,2] -> C[1,2]
    # t2 = 5: A[1,1]*B[1,2] -> C[1,2]
    # Wait, that's missing C[1,0] and C[1,1]. Let me redo.
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # 
    # That's 12 terms for schoolbook, not 6.
    # 
    # Can we do M_{2,2,3} in rank 6? Let me think...
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]  (row 0 of C)
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]  (row 1 of C)
    # 
    # In tensor form, a rank-1 term for M_{2,2,3} is:
    # U[i0,k0] * V[k0,j0] * W[i0,j0]
    # This contributes to C[i0,j0] with coefficient U[i0,k0]*V[k0,j0]*W[i0,j0]
    # 
    # For the term A[0,0]*B[0,j0] contributing to C[0,j0]:
    # We need U[0,0]=1, V[0,j0]=1 for all j0, W[0,j0]=1 for all j0.
    # But a rank-1 tensor has fixed U, V, W matrices. We can't have V[0,j0]=1 for all j0
    # and also have other terms work.
    # 
    # Actually, a rank-1 tensor T = u ⊗ v ⊗ w means T[i,k,j] = u[i,k]*v[k,j]*w[i,j].
    # For M_{2,2,3}, i in {0,1}, k in {0,1}, j in {0,1,2}.
    # A rank-1 term with u[0,0]=1, v[0,0]=1, v[0,1]=1, v[0,2]=1, w[0,0]=1, w[0,1]=1, w[0,2]=1
    # would give T[0,0,0]=1, T[0,0,1]=1, T[0,0,2]=1, and 0 elsewhere.
    # This corresponds to A[0,0] multiplying all of B[0,:] to give C[0,:].
    # 
    # So we can have:
    # Term 1: u[0,0]=1, v[0,:]=1, w[0,:]=1 -> A[0,0]*B[0,:] -> C[0,:]
    # Term 2: u[0,1]=1, v[1,:]=1, w[0,:]=1 -> A[0,1]*B[1,:] -> C[0,:]
    # Term 3: u[1,0]=1, v[0,:]=1, w[1,:]=1 -> A[1,0]*B[0,:] -> C[1,:]
    # Term 4: u[1,1]=1, v[1,:]=1, w[1,:]=1 -> A[1,1]*B[1,:] -> C[1,:]
    # 
    # That's only 4 terms! And it correctly computes C = A*B for 2x2 * 2x3.
    # Let me verify: C[0,0] = Term1[0,0] + Term2[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] ✓
    # C[0,1] = Term1[0,1] + Term2[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] ✓
    # C[0,2] = Term1[0,2] + Term2[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] ✓
    # C[1,0] = Term3[1,0] + Term4[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] ✓
    # C[1,1] = Term3[1,1] + Term4[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] ✓
    # C[1,2] = Term3[1,2] + Term4[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] ✓
    # 
    # So rank(M_{2,2,3}) <= 4! And the lower bound is max(2, 2*3/2, 2*3/2) = 3, so rank could be 3 or 4.
    # I'll use rank 4.
    # 
    # Total rank: 7 * 4 = 28.
    
    R = 28
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Inner schoolbook for M_{2,2,3}: rank 4
    # t2=0: A[0,0]*B[0,:] -> C[0,:]
    # t2=1: A[0,1]*B[1,:] -> C[0,:]
    # t2=2: A[1,0]*B[0,:] -> C[1,:]
    # t2=3: A[1,1]*B[1,:] -> C[1,:]
    
    col = 0
    for s in range(7):
        for t2 in range(4):
            # Determine inner indices
            if t2 == 0:
                i0, k0 = 0, 0  # A[0,0] * B[0,:] -> C[0,:]
            elif t2 == 1:
                i0, k0 = 0, 1  # A[0,1] * B[1,:] -> C[0,:]
            elif t2 == 2:
                i0, k0 = 1, 0  # A[1,0] * B[0,:] -> C[1,:]
            else:
                i0, k0 = 1, 1  # A[1,1] * B[1,:] -> C[1,:]
            
            # U: index = (i1*2+i0)*4 + (k1*2+k0)
            for r in range(4):
                if alpha[s][r] != 0:
                    i1, k1 = r // 2, r % 2
                    u_idx = (i1 * 2 + i0) * 4 + (k1 * 2 + k0)
                    U[u_idx, col] += alpha[s][r]
            
            # V: index = (k1*2+k0)*6 + (j1*3+j0)
            # V[k0, j0] = 1 for all j0
            for c in range(4):
                if beta[s][c] != 0:
                    k1, j1 = c // 2, c % 2
                    for j0 in range(3):
                        v_idx = (k1 * 2 + k0) * 6 + (j1 * 3 + j0)
                        V[v_idx, col] += beta[s][c]
            
            # W: index = (j1*3+j0)*4 + (i1*2+i0)
            # W[i0, j0] = 1 for all j0
            for a, b, gamma in recon[s]:
                # a = i1, b = j1
                for j0 in range(3):
                    w_idx = (b * 3 + j0) * 4 + (a * 2 + i0)
                    W[w_idx, col] += gamma
            
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
