import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strategy: Use Kronecker product structure
    # M_{2,4,4} can be decomposed using the fact that:
    # A (2x4) * B (4x4) = C (2x4)
    # 
    # View A as 2x2 blocks of 1x2: A = [A_top; A_bot] where A_top, A_bot are 1x4
    # View B as 2x2 blocks of 2x2: B = [[B00,B01],[B10,B11]]
    #
    # C_top = A_top * B = A_top[:2]*B00 + A_top[2:]*B10 (1x4)
    # C_bot = A_bot * B = A_bot[:2]*B01 + A_bot[2:]*B11 (1x4)
    #
    # Actually, let's use a cleaner decomposition:
    # Think of it as: for each of 4 output columns j, C[:,j] = A * B[:,j]
    # B[:,j] is 4x1, A is 2x4, so C[:,j] is 2x1
    # C[:,j] = A * B[:,j] = sum_k A[:,k]*B[k,j]
    #
    # Better structural approach: Use the rank-7 Strassen for 2x2x2
    # and combine with a rank-3 scheme for the 1x2x2 part
    #
    # M_{2,4,4}: Let's partition differently
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [C1 | C2] where C1, C2 are 2x2
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    #
    # This is 4 matrix products of 2x2 matrices, but with sharing.
    # If we use Strassen for each, that's 4*7 = 28.
    # But we can share: the Strassen intermediates for A1 appear in both C1 and C2.
    #
    # Let me try: compute Strassen intermediates for A1 once, reuse for C1 and C2.
    # Similarly for A2.
    #
    # Strassen for 2x2: 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # For C1 = A1*B11 + A2*B21:
    #   = (A1*B11) + (A2*B21)
    #   Each 2x2*2x2 needs 7 mults via Strassen, total 14
    # For C2 = A1*B12 + A2*B22:
    #   = (A1*B12) + (A2*B22)
    #   Another 14 mults
    # Total: 28. Same as before.
    #
    # To get rank 24, I need to find 4 shared multiplications.
    #
    # Alternative: Use a different known result.
    # rank(M_{2,4,4}) = 24 is known. Let me construct it explicitly.
    #
    # The construction: 
    # Use the fact that M_{2,4,4} has a decomposition of rank 24.
    # One way: use 3 copies of a rank-8 scheme plus 8 shared terms.
    #
    # Actually, let me try a direct rank-24 construction.
    # The key: partition into 2x2 blocks and use optimized Strassen-like.
    #
    # Let me try: for each of the 4 block products (A_i * B_j -> C_k),
    # use a rank-6 scheme instead of rank-7, saving 4 total.
    # But rank-6 for 2x2x2 is not known to exist (lower bound is 7).
    #
    # So the saving must come from sharing across blocks.
    #
    # Let me try a different partition:
    # A is 2x4. Split columns into 4 groups of 1: A[:,0], A[:,1], A[:,2], A[:,3]
    # B is 4x4. 
    # C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # For fixed i (row of A), C[i,:] = A[i,:] * B (1x4 times 4x4 = 1x4)
    # This is a 1x4 * 4x4 product.
    # Rank of 1x4*4x4 tensor: each output C[i,j] = sum_k A[i,k]*B[k,j]
    # For fixed i, we have 4 outputs, each is a dot product of length 4.
    # Total rank for one i: at most 4*4=16 schoolbook, but can be done in 4
    # if we precompute A[i,k]*B[k,j]... no that's the same.
    #
    # Actually for 1x4 * 4x4: C[j] = sum_k a_k * B[k,j] for j=0,1,2,3
    # This is just multiplying a 4-vector by a 4x4 matrix.
    # The tensor rank is 4 (each column of B times the vector a).
    # Wait no: C[j] = a_0*B[0,j] + a_1*B[1,j] + a_2*B[2,j] + a_3*B[3,j]
    # The bilinear form has rank 4 (one per k, shared across j).
    # So for each row i of A, we need rank 4 to compute C[i,:].
    # Total for 2 rows: rank 8? That can't be right because the B entries are shared.
    #
    # Wait, the tensor is T[i,j,k] = delta (the identity tensor for matmul).
    # The rank is the minimum number of rank-1 tensors to sum to T.
    # For M_{2,4,4}, T[i,j,k] = 1 if... no, T is defined by:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the 
    # appropriate sense). No wait, that's not right either.
    #
    # The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i in [n], 
    # j in [m], k in [p]... no. The tensor is defined such that:
    # <T, A ⊗ B> = C where C[i,j] = sum_k A[i,k]*B[k,j]
    # So T[i,j,k] = 1 for all valid i,j,k. It's the all-ones tensor!
    # No, that's not right. Let me be precise.
    #
    # The multiplication tensor T_{n,m,p} ∈ R^{n*m × m*p × p*n} (or similar indexing)
    # is defined by T_{(i,k),(k,j),(i,j)} = 1 for all i,j,k.
    # With the given indexing: U has n*m rows (A elements), V has m*p rows (B elements),
    # W has p*n rows (C elements, indexed as k*n+i).
    #
    # So T[(i*4+j), (j*4+k), (k*2+i)] = 1 for all i in [0,2), j in [0,4), k in [0,4).
    #
    # The rank is the minimum r such that T = sum_{r'} U_r' ⊗ V_r' ⊗ W_r'
    # where U_r' ∈ R^{8}, V_r' ∈ R^{16}, W_r' ∈ R^{8}.
    #
    # For the all-ones pattern T[a,b,c] = 1 when a=i*4+j, b=j*4+k, c=k*2+i,
    # the rank is known to be 24 for M_{2,4,4}.
    #
    # Let me just verify rank 28 works correctly and return it, since I'm not
    # confident in a rank-24 construction from scratch.
    #
    # Actually, let me try to get rank 24 using a known construction.
    # The idea from Landsberg et al.: use the "little" matrix multiplication.
    #
    # Alternative approach: use the fact that 2x4*4x4 can be seen as
    # (2x2 * 2x2) ⊗ (1x2 * 2x2) in tensor product sense.
    # rank(M_{2,2,2}) = 7, rank(M_{1,2,2}) = 2 (it's just 2 dot products)
    # Tensor product rank: 7*2 = 14? That's too optimistic.
    #
    # Actually M_{1,2,2}: C[0,j] = sum_k A[0,k]*B[k,j] for j=0,1
    # T[a,b,c] where a in [0,2), b in [0,4), c in [0,2)
    # T[k, j*2+0... wait let me re-index.
    # For M_{1,2,2}: n=1, m=2, p=2
    # U: 1*2=2 rows (A[0,0], A[0,1])
    # V: 2*2=4 rows (B[0,0], B[0,1], B[1,0], B[1,1])
    # W: 2*1=2 rows (C[0,0], C[0,1]) indexed as k*n+i = k*1+0 = k
    # T[A_idx, B_idx, C_idx]: T[k, j*2+k... 
    # A_idx = i*m+k = 0*2+k = k
    # B_idx = k*p+j = k*2+j
    # C_idx = j*n+i = j*1+0 = j
    # So T[k, 2k+j, j] = 1 for k in {0,1}, j in {0,1}
    # This has 4 nonzero entries. Rank is 2 (two rank-1 terms: one for k=0, one for k=1).
    #
    # So M_{2,4,4} might relate to M_{2,2,2} ⊗ M_{1,2,2} with rank 7*2 = 14?
    # But that's not how tensor products work for matmul.
    #
    # Let me just go with rank 28 which I know works, and make sure it's correct.
    
    def get_strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen, each 4x7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A = [[A00, A01], [A10, A11]], indexed as [A00, A01, A10, A11] = [0,1,2,3]
        # B = [[B00, B01], [B10, B11]], indexed as [B00, B01, B10, B11] = [0,1,2,3]
        # C = [[C00, C01], [C10, C11]], indexed as [C00, C01, C10, C11] = [0,1,2,3]
        #
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M5 + M7
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # contributes to C00 and C11
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01 += M3
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1  # C00 -= M4, C10 += M4
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1  # C01 += M5, C11 += M5
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1  # C00 += M6
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1  # C00 -= M7, C11 += M7
        
        return U, V, W
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Verify Strassen 2x2x2
    T2 = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U2[i*2+k, r] * V2[k*2+j, r] * W2[j*2+i, r]
                T2[i,j,k] = s
    if not np.allclose(T2, 1.0):
        # Fallback to schoolbook
        rank = 32
        U_fb = np.zeros((8, rank))
        V_fb = np.zeros((16, rank))
        W_fb = np.zeros((8, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+k, col] = 1
                    V_fb[k*4+j, col] = 1
                    W_fb[j*2+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now build rank-28 for 2x4x4
    rank = 28
    U_full = np.zeros((8, rank))
    V_full = np.zeros((16, rank))
    W_full = np.zeros((8, rank))
    
    # A is 2x4: A[i,k] indexed as i*4+k in U
    # B is 4x4: B[k,j] indexed as k*4+j in V
    # C is 2x4: C[i,j] indexed as j*2+i in W (per spec: W[k*n+i,r] where k is col, i is row)
    # Wait, spec says W[k*n+i,r] where the output is C with dimensions... 
    # C is 2x4 (n x p where n=2 rows, p=4 cols)
    # W has p*n = 4*2 = 8 rows, indexed as k*n+i where k is column (0-3), i is row (0-1)
    # So W_idx = k*2 + i for C[i,k]
    
    # Block decomposition:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Split k into k1 in {0,1} and k2 in {2,3}
    # C[i,j] = sum_{k1=0}^1 A[i,k1]*B[k1,j] + sum_{k2=0}^1 A[i,2+k2]*B[2+k2,j]
    #
    # For each i and j, this is two 2x2x2-like products.
    #
    # Better: group into 2x2 blocks.
    # Let A = [A_L | A_R] where A_L = A[:,0:2] (2x2), A_R = A[:,2:4] (2x2)
    # Let B = [[B_TL, B_TR], [B_BL, B_BR]] (each 2x2)
    # Let C = [C_L | C_R] where C_L = C[:,0:2] (2x2), C_R = C[:,2:4] (2x2)
    #
    # C_L = A_L * B_TL + A_R * B_BL
    # C_R = A_L * B_TR + A_R * B_BR
    #
    # 4 block products, each 2x2*2x2. Use Strassen for each: 4*7 = 28.
    
    # Block 0: A_L * B_TL -> contributes to C_L
    # A_L = A[0:2, 0:2], indices in U: 0,1,2,3 (A[0,0],A[0,1],A[1,0],A[1,1])
    # B_TL = B[0:2, 0:2], indices in V: 0,1,2,3 (B[0,0],B[0,1],B[1,0],B[1,1])
    # C_L = C[0:2, 0:2], indices in W: j*2+i for i in {0,1}, j in {0,1}: 0,1,2,3
    
    U_full[0:4, 0:7] = U2
    V_full[0:4, 0:7] = V2
    W_full[0:4, 0:7] = W2
    
    # Block 1: A_R * B_BL -> contributes to C_L
    # A_R = A[0:2, 2:4], indices in U: 4,5,6,7 (A[0,2],A[0,3],A[1,2],A[1,3])
    # B_BL = B[2:4, 0:2], indices in V: 8,9,10,11 (B[2,0],B[2,1],B[3,0],B[3,1])
    # C_L indices in W: 0,1,2,3
    
    U_full[4:8, 7:14] = U2
    V_full[8:12, 7:14] = V2
    W_full[0:4, 7:14] = W2
    
    # Block 2: A_L * B_TR -> contributes to C_R
    # A_L indices in U: 0,1,2,3
    # B_TR = B[0:2, 2:4], indices in V: 4,5,6,7 (B[0,2],B[0,3],B[1,2],B[1,3])
    # C_R = C[0:2, 2:4], indices in W: j*2+i for i in {0,1}, j in {2,3}: 4,5,6,7
    
    U_full[0:4, 14:21] = U2
    V_full[4:8, 14:21] = V2
    W_full[4:8, 14:21] = W2
    
    # Block 3: A_R * B_BR -> contributes to C_R
    # A_R indices in U: 4,5,6,7
    # B_BR = B[2:4, 2:4], indices in V: 12,13,14,15 (B[2,2],B[2,3],B[3,2],B[3,3])
    # C_R indices in W: 4,5,6,7
    
    U_full[4:8, 21:28] = U2
    V_full[12:16, 21:28] = V2
    W_full[4:8, 21:28] = W2
    
    # Verify
    T = np.zeros((2, 4, 4))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for r in range(28):
                    s += U_full[i*4+k, r] * V_full[k*4+j, r] * W_full[j*2+i, r]
                T[i,j,k] = s
    
    if np.allclose(T, 1.0):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook rank 32
    rank = 32
    U_fb = np.zeros((8, rank))
    V_fb = np.zeros((16, rank))
    W_fb = np.zeros((8, rank))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U_fb[i*4+k, col] = 1
                V_fb[k*4+j, col] = 1
                W_fb[j*2+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
