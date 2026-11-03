import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Rank 24 decomposition for M_{2,4,4}
    # C (2x4) = A (2x4) * B (4x4)
    # Partition: A = [A0, A1] where A0=A[0,:], A1=A[1,:]
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C0 = A0[:,0:2]*B00 + A0[:,2:4]*B10  (1x4 row)
    # C1 = A1[:,0:2]*B01 + A1[:,2:4]*B11  (1x4 row)
    
    # Actually, let me use the correct partitioning:
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
    # C is 2x4: rows 0,1; cols 0,1,2,3
    
    # C[0,0:2] = A[0,0:2]*B[0:2,0:2] + A[0,2:4]*B[2:4,0:2]
    # C[0,2:4] = A[0,0:2]*B[0:2,2:4] + A[0,2:4]*B[2:4,2:4]
    # C[1,0:2] = A[1,0:2]*B[0:2,0:2] + A[1,2:4]*B[2:4,0:2]
    # C[1,2:4] = A[1,0:2]*B[0:2,2:4] + A[1,2:4]*B[2:4,2:4]
    
    # So we have 4 sub-problems of 1x2 * 2x2 = 1x2, but that's not standard.
    
    # Better: think of it as 4 sub-problems of 2x1 * 2x2 or similar.
    
    # Actually the cleanest: M_{2,4,4} = M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2}
    # But with sharing, we can get rank 24.
    
    # Let me use a known rank-24 construction.
    # The idea: use 6 Strassen-like decompositions with shared terms.
    
    # U: 8 x 24 (A has 8 elements)
    # V: 16 x 24 (B has 16 elements)  
    # W: 8 x 24 (C has 8 elements, indexed as k*n+i = k*2+i)
    
    rank = 24
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # The 4 blocks correspond to:
    # Block 0: C[0,0:2] from A[0,0:2]*B[0:2,0:2]
    # Block 1: C[0,2:4] from A[0,0:2]*B[0:2,2:4]
    # Block 2: C[1,0:2] from A[1,0:2]*B[2:4,0:2]
    # Block 3: C[1,2:4] from A[1,0:2]*B[2:4,2:4]
    
    # Wait, that's not right. Let me redo:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # For i=0, j=0: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # For i=0, j=1: A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
    # etc.
    
    # Partition into 2x2 blocks of B:
    # B[0:2,0:2] = B00, B[0:2,2:4] = B01
    # B[2:4,0:2] = B10, B[2:4,2:4] = B11
    
    # C[0,0:2] = A[0,0:2]*B00 + A[0,2:4]*B10  (1x2)
    # C[0,2:4] = A[0,0:2]*B01 + A[0,2:4]*B11  (1x2)
    # C[1,0:2] = A[1,0:2]*B00 + A[1,2:4]*B10  (1x2)
    # C[1,2:4] = A[1,0:2]*B01 + A[1,2:4]*B11  (1x2)
    
    # Each is a 1x2 * 2x2 product, which has rank 4 (schoolbook) or can be done with fewer.
    # Actually 1x2 * 2x2: output is 1x2, so 2 outputs.
    # C[j] = a0*B0j + a1*B1j for j=0,1
    # This is just 2 separate 1x2*2x1 products, each rank 2, total rank 4 per block.
    # 4 blocks * 4 = 16? That seems too low. Let me reconsider.
    
    # Actually no. A[0,0:2] is a 1x2 vector, B00 is 2x2.
    # A[0,0:2]*B00 gives a 1x2 vector: [a0*B00[0,0]+a1*B00[1,0], a0*B00[0,1]+a1*B00[1,1]]
    # This requires 4 multiplications (a0*B00[0,0], a0*B00[0,1], a1*B00[1,0], a1*B00[1,1])
    # But we can share: the products a0*B00[*,*] are shared between computing C[0,0:2] and... 
    # Actually for block 0, we compute A[0,0:2]*B00, which gives 2 outputs using 4 mults.
    # For block 2, we compute A[1,0:2]*B00, which gives 2 outputs using 4 mults.
    # These don't share because A[0,0:2] != A[1,0:2].
    
    # So each block needs rank 4, total 16? But that contradicts known results.
    
    # Let me look at this differently. The tensor is M_{2,4,4}.
    # Known: rank(M_{2,4,4}) = 24.
    
    # Construction via "border rank" or direct sum with sharing.
    # Actually, M_{2,4,4} = M_{2,2,2} ⊗ M_{1,2,2} in some sense?
    
    # Let me just use the rank-28 Strassen and see if I can improve.
    # Or use a different known construction.
    
    # After research: rank(M_{2,4,4}) = 24 can be achieved.
    # The construction uses the fact that we can write:
    # C = A*B where we group terms cleverly.
    
    # Let me try a direct construction with rank 24.
    
    # U rows: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]
    # V rows: B[0,0], B[0,1], B[0,2], B[0,3], B[1,0], B[1,1], B[1,2], B[1,3],
    #         B[2,0], B[2,1], B[2,2], B[2,3], B[3,0], B[3,1], B[3,2], B[3,3]
    # W rows: C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]
    #         (indexed as k*n+i: k=0->C[0,*], k=1->C[1,*])
    
    # I'll use a known rank-24 pattern.
    # The key insight: use 3 Strassen decompositions for the 4 2x2 blocks,
    # with one term shared between blocks.
    
    # Actually, let me just verify rank 28 works and return that,
    # since finding rank 24 requires careful construction.
    
    # Let me construct rank 28 via 4 independent Strassen 2x2x2 decompositions.
    
    def get_strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen, each 4x7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    U_full = np.zeros((8, 28))
    V_full = np.zeros((16, 28))
    W_full = np.zeros((8, 28))
    
    # Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A rows 0:4 (A[0,0],A[0,1],A[1,0],A[1,1]), B rows 0:4, C rows 0:4
    U_full[0:4, 0:7] = U2
    V_full[0:4, 0:7] = V2
    W_full[0:4, 0:7] = W2
    
    # Block 1: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # A rows 4:8 (A[0,2],A[0,3],A[1,2],A[1,3]), B rows 8:12, C rows 0:4
    U_full[4:8, 7:14] = U2
    V_full[8:12, 7:14] = V2
    W_full[0:4, 7:14] = W2
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # A rows 0:4, B rows 4:8, C rows 4:8
    U_full[0:4, 14:21] = U2
    V_full[4:8, 14:21] = V2
    W_full[4:8, 14:21] = W2
    
    # Block 3: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # A rows 4:8, B rows 12:16, C rows 4:8
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
                    s += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*2+i, r]
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
                U_fb[i*4+j, col] = 1
                V_fb[j*4+k, col] = 1
                W_fb[k*2+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
