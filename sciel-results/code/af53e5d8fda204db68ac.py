import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Approach: Split middle dimension into two halves of size 2.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is 2x2 @ 2x8 -> 2x8
    # For 2x2 @ 2x8, use rank-7 Strassen-like decomposition per column-pair
    # but actually we need a decomposition for 2x2 @ 2x8.
    # 
    # Use the fact that 2x2 @ 2x8 can be done with 7 rank-1 terms by
    # applying Strassen's 7 multiplications to the 2x2 structure,
    # where each "scalar" multiplication involves a row of B (length 8).
    # This gives rank 7 for 2x2 @ 2x8, total rank = 2 * 7 = 14.
    #
    # Verification: For 2x2 @ 2x2, Strassen uses 7 rank-1 tensors.
    # For 2x2 @ 2x8, we can use the same 7 rank-1 tensors but with
    # B vectors of length 8 instead of 2. This is valid because the
    # tensor structure is the same - we're just extending B.
    #
    # Actually, this is NOT correct. Strassen's 7 terms for 2x2@2x2
    # involve specific linear combinations of B entries. For 2x2@2x8,
    # we need to handle 8 columns, so we'd need 7*4=28 terms if we
    # apply Strassen to each 2-column block.
    #
    # Let me use a different approach: direct rank-4 for 2x2@2x8
    # (naive: 4 multiplications per row of A against all of B)
    # Total rank = 2 * 4 = 8? Let's verify.
    #
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # For fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is a linear combination of rows of B, weighted by A[i,:].
    # In tensor form: T = sum_{i,k} A[i,k] * B[k,:] * e_i
    # Rank <= n*m = 2*4 = 8? No, that's the number of terms in the
    # expansion, but each term is rank-1 in the tensor sense.
    #
    # Actually: T[i,j,k] = delta(i,k) for the multiplication tensor? No.
    # The multiplication tensor M_{n,m,p} has M_{i,j,k} = 1 if and only if
    # C[i,j] = sum_k A[i,k]B[k,j]. So M_{i,j,k} = 1 for all valid i,j,k.
    # Wait no, that's the all-ones tensor which has rank 1.
    #
    # I'm confusing myself. Let me just use the verified rank-56 solution.
    
    # Strassen decomposition for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1: (A11+A22)*(B11+B22) -> C11, C22
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # M2: (A21+A22)*B11 -> C21, C22
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # M3: A11*(B12-B22) -> C12, C22
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # M4: A22*(B21-B11) -> C11, C21
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # M5: (A11+A12)*B22 -> C11, C12
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # M6: (A21-A11)*(B11+B12) -> C22
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: (A12-A22)*(B21+B22) -> C11
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j_group in range(2):
        a_map = [
            0 * 4 + j_group * 2,
            0 * 4 + j_group * 2 + 1,
            1 * 4 + j_group * 2,
            1 * 4 + j_group * 2 + 1
        ]
        
        for k_group in range(4):
            b_map = [
                (j_group * 2) * 8 + k_group * 2,
                (j_group * 2) * 8 + k_group * 2 + 1,
                (j_group * 2 + 1) * 8 + k_group * 2,
                (j_group * 2 + 1) * 8 + k_group * 2 + 1
            ]
            
            c_map = [
                (k_group * 2) * 2 + 0,
                (k_group * 2 + 1) * 2 + 0,
                (k_group * 2) * 2 + 1,
                (k_group * 2 + 1) * 2 + 1
            ]
            
            base_r = (j_group * 4 + k_group) * 7
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
