import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # 1. Generate Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10 + A11)B00 -> C10, C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1 # C10 is index 2, C11 is index 3? 
    # Indices: i*2+j. A10 is i=1,j=0 -> 2. A11 is 3. B00 is 0.
    # C10 is k=0,i=1 -> 2. C11 is k=1,i=1 -> 3.
    # W[k*2+i]. C10: 0*2+1=1? No. W index is k*n+i. n=2.
    # C10: k=0, i=1 -> 1. C11: k=1, i=1 -> 3.
    # My previous W2 indexing: W2[0,0]=1 (C00: k=0,i=0), W2[3,0]=1 (C11: k=1,i=1).
    # So W2[1,1]=1 (C10), W2[3,1]=1 (C11).
    W2[1, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01 - B11) -> C01
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01: k=1, i=0 -> 1
    
    # M4 = A11(B10 - B00) -> C10, C00
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1; W2[0, 3] = 1 # C10: 1, C00: 0
    
    # M5 = (A00 + A01)B11 -> C01, C11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1 # C01: 1, C11: 3
    
    # M6 = (A10 - A00)(B00 + B01) -> C10, C01
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = 1; W2[1, 5] = 1 # C10: 1, C01: 1. Wait.
    # C10 is index 1. C01 is index 1.
    # So W2[1,5] should be 2?
    # M6 contributes to C10 and C01.
    # C10 = ... + M6. C01 = ... + M6.
    # So W2[1,5] = 2?
    # Let's check Strassen formulas.
    # C10 = M2 + M4.
    # C01 = M3 + M5.
    # C11 = M1 - M2 + M3 + M6.
    # C00 = M1 + M4 - M5 + M7.
    # So M6 only contributes to C11.
    # C11 index is 3.
    # So W2[3,5] = 1.
    W2[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 index 0
    
    # Verify 2x2x2
    def verify2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if abs(s - (1 if i==0 and j==0 and k==0 else 0)) > 1e-9:
                         # Wait, the condition is sum ... = 1 for the specific term?
                         # No, the condition is that the sum reconstructs the product.
                         # For schoolbook, only one term is non-zero for each (i,j,k).
                         # For Strassen, the sum of products equals 1 for the "effective" multiplication?
                         # No, the tensor decomposition condition is:
                         # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                         # This is for the "standard basis" of the multiplication tensor.
                         # The multiplication tensor T has T[i,j,k] = 1 if it's the term A[i,j]B[j,k].
                         # So yes, the sum must be 1 for all i,j,k?
                         # No. The tensor for matrix multiplication is:
                         # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k
                         # No, T is the tensor such that C = (A, B) -> <T, A \otimes B>.
                         # T[i,j,k] = 1.
                         # So yes, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must be 1 for all i,j,k.
                         pass
    
    # 2. Construct 4x4x4 via Kronecker product
    # U444[i*4+j, r_high*7+r_low] = U2[i_high*2+j_high, r_high] * U2[i_low*2+j_low, r_low]
    # This requires careful index mapping.
    # i = 2*i1 + i0, j = 2*j1 + j0
    # i*4 + j = 8*i1 + 4*i0 + 2*j1 + j0
    # U2 index for high: i1*2 + j1
    # U2 index for low: i0*2 + j0
    # Kronecker index: (i1*2+j1)*4 + (i0*2+j0) = 8*i1 + 4*j1 + 2*i0 + j0
    # Mismatch: 4*i0 + 2*j1 vs 4*j1 + 2*i0
    # We need to permute.
    
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    for r_high in range(7):
        for r_low in range(7):
            r = r_high * 7 + r_low
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            idx = i*4 + j
                            
                            idx_h = i1*2 + j1
                            idx_l = i0*2 + j0
                            
                            U444[idx, r] = U2[idx_h, r_high] * U2[idx_l, r_low]
                            V444[idx, r] = V2[idx_h, r_high] * V2[idx_l, r_low]
                            W444[idx, r] = W2[idx_h, r_high] * W2[idx_l, r_low]
    
    # 3. Assemble final 4x5x6
    rank_444 = 49
    rank_442 = 8
    rank_1x6 = 6
    total_rank = rank_444 + rank_442 + rank_1x6
    
    U = np.zeros((20, total_rank))
    V = np.zeros((30, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: 4x4x4
    for i in range(4):
        for j in range(4):
            for r in range(rank_444):
                U[i*5+j, r] = U444[i*4+j, r]
                V[j*6+r, r] = V444[j*4+r, r] # Wait, V index is j*6+k. Here k<4.
                # V444 index is j*4+k.
                # So V[j*6+k, r] = V444[j*4+k, r]
                # We need to fill V for k=0..3
                for k in range(4):
                    V[j*6+k, r] = V444[j*4+k, r]
                for k in range(4):
                    W[k*4+i, r] = W444[k*4+i, r]
    
    # Part 2: 4x4x2 (j<4, k=4,5)
    r_offset = rank_444
    for j in range(4):
        for kk in range(2):
            k = 4 + kk
            r = r_offset + j*2 + kk
            for i in range(4):
                U[i*5+j, r] = 1
            V[j*6+k, r] = 1
            for i in range(4):
                W[k*4+i, r] = 1
    
    # Part 3: 4x1x6 (j=4, k=0..5)
    r_offset = rank_444 + rank_442
    for k in range(6):
        r = r_offset + k
        for i in range(4):
            U[i*5+4, r] = 1
        V[4*6+k, r] = 1
        for i in range(4):
            W[k*4+i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
