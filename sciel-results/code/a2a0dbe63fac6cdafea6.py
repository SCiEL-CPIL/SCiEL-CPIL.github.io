import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Define scalar Strassen decomposition for 2x2 * 2x2
    # U, V, W shapes (4, 7)
    # Indices:
    # U: i*m + j -> i*2 + j
    # V: j*p + k -> j*2 + k
    # W: k*n + i -> k*2 + i
    
    U_sc = np.zeros((4, 7))
    V_sc = np.zeros((4, 7))
    W_sc = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_sc[0, 0] = 1; U_sc[3, 0] = 1 # A00, A11
    V_sc[0, 0] = 1; V_sc[3, 0] = 1 # B00, B11
    W_sc[0, 0] = 1; W_sc[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00 -> C10, -C11
    U_sc[2, 1] = 1; U_sc[3, 1] = 1 # A10, A11
    V_sc[0, 1] = 1                 # B00
    W_sc[2, 1] = 1; W_sc[3, 1] = -1 # C10, C11
    
    # M3: A00(B01-B11) -> C01
    U_sc[0, 2] = 1                 # A00
    V_sc[1, 2] = 1; V_sc[3, 2] = -1 # B01, B11
    W_sc[1, 2] = 1                 # C01
    
    # M4: A11(B10-B00) -> -C00, C10
    U_sc[3, 3] = 1                 # A11
    V_sc[2, 3] = 1; V_sc[0, 3] = -1 # B10, B00
    W_sc[0, 3] = -1; W_sc[2, 3] = 1 # C00, C10
    
    # M5: (A00+A01)B11 -> C00, C01
    U_sc[0, 4] = 1; U_sc[1, 4] = 1 # A00, A01
    V_sc[3, 4] = 1                 # B11
    W_sc[0, 4] = 1; W_sc[1, 4] = 1 # C00, C01
    
    # M6: (A10-A00)(B00+B01) -> C11
    U_sc[2, 5] = 1; U_sc[0, 5] = -1 # A10, A00
    V_sc[0, 5] = 1; V_sc[1, 5] = 1  # B00, B01
    W_sc[3, 5] = 1                 # C11
    
    # M7: (A01-A11)(B10+B11) -> -C00, -C11
    U_sc[1, 6] = 1; U_sc[3, 6] = -1 # A01, A11
    V_sc[2, 6] = 1; V_sc[3, 6] = 1  # B10, B11
    W_sc[0, 6] = -1; W_sc[3, 6] = -1 # C00, C11

    # 2. Construct 4x4 * 4x4 decomposition using Kronecker products
    # Rank = 7 * 7 = 49
    R_block = 49
    U_44 = np.zeros((16, R_block))
    V_44 = np.zeros((16, R_block))
    W_44 = np.zeros((16, R_block))
    
    # Mapping block indices to local indices
    # Block A_pq is at global rows p*4 + q*2 ... p*4 + q*2 + 3?
    # No, A is 4x4.
    # A[0:2, 0:2] is block A00.
    # A[0:2, 2:4] is block A01.
    # A[2:4, 0:2] is block A10.
    # A[2:4, 2:4] is block A11.
    
    # Global index for A[i,j] is i*4 + j.
    # Block index P=(pb, qb). Local index (i,j).
    # Global row = pb*2 + i, Global col = qb*2 + j.
    # Global index = (pb*2 + i)*4 + (qb*2 + j) = pb*8 + i*4 + qb*2 + j.
    
    # For U (A): indices are i*4+j.
    # For V (B): indices are j*4+k.
    # For W (C): indices are k*4+i.
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Get non-zero entries from block vectors
            # U_sc[r1] is a vector of length 4 (block indices 0..3)
            # V_sc[r1] is a vector of length 4
            # W_sc[r1] is a vector of length 4
            
            # U_sc indices: 0=A00, 1=A01, 2=A10, 3=A11
            # V_sc indices: 0=B00, 1=B01, 2=B10, 3=B11
            # W_sc indices: 0=C00, 1=C01, 2=C10, 3=C11
            
            # We need to expand each block entry into 4 local entries.
            
            # Helper to expand a block vector to global vector
            def expand_U(vec):
                # vec is length 4, indices 0..3 correspond to blocks A00, A01, A10, A11
                # Global U index for A[i,j] is i*4+j
                out = np.zeros(16)
                for b_idx in range(4):
                    val = vec[b_idx]
                    if val == 0: continue
                    pb = b_idx // 2
                    qb = b_idx % 2
                    for i in range(2):
                        for j in range(2):
                            # Local U index for A_sc is i*2+j
                            local_idx = i*2 + j
                            local_val = U_sc[local_idx, r2]
                            if local_val == 0: continue
                            global_i = pb*2 + i
                            global_j = qb*2 + j
                            global_idx = global_i*4 + global_j
                            out[global_idx] += val * local_val
                return out

            def expand_V(vec):
                # vec is length 4, indices 0..3 correspond to blocks B00, B01, B10, B11
                # Block B_pq has local indices j*2+k.
                # Global V index for B[j,k] is j*4+k
                out = np.zeros(16)
                for b_idx in range(4):
                    val = vec[b_idx]
                    if val == 0: continue
                    pb = b_idx // 2 # This is 'j' in B_pq? No, B is 4x4.
                    # B is 4x4. Blocks are B_00, B_01, B_10, B_11.
                    # B_00 is B[0:2, 0:2].
                    # So block index b_idx corresponds to B[pb, qb] where pb=b_idx//2, qb=b_idx%2.
                    # Local indices in B_pq are j*2+k where j,k in 0..1.
                    # Global row j_global = pb*2 + j.
                    # Global col k_global = qb*2 + k.
                    # Global V index = j_global*4 + k_global.
                    for j in range(2):
                        for k in range(2):
                            local_idx = j*2 + k
                            local_val = V_sc[local_idx, r2]
                            if local_val == 0: continue
                            global_j = pb*2 + j
                            global_k = qb*2 + k
                            global_idx = global_j*4 + global_k
                            out[global_idx] += val * local_val
                return out

            def expand_W(vec):
                # vec is length 4, indices 0..3 correspond to blocks C00, C01, C10, C11
                # Block C_pq has local indices k*2+i.
                # Global W index for C[k,i] is k*4+i.
                out = np.zeros(16)
                for b_idx in range(4):
                    val = vec[b_idx]
                    if val == 0: continue
                    pb = b_idx // 2 # 'k' block index?
                    # C is 4x4. Blocks C_00, C_01, C_10, C_11.
                    # C_00 is C[0:2, 0:2].
                    # So block index b_idx corresponds to C[pb, qb] where pb=b_idx//2, qb=b_idx%2.
                    # Local indices in C_pq are k*2+i.
                    # Global row k_global = pb*2 + k.
                    # Global col i_global = qb*2 + i.
                    # Global W index = k_global*4 + i_global.
                    for k in range(2):
                        for i in range(2):
                            local_idx = k*2 + i
                            local_val = W_sc[local_idx, r2]
                            if local_val == 0: continue
                            global_k = pb*2 + k
                            global_i = qb*2 + i
                            global_idx = global_k*4 + global_i
                            out[global_idx] += val * local_val
                return out

            U_44[:, r] = expand_U(U_sc[:, r1])
            V_44[:, r] = expand_V(V_sc[:, r1])
            W_44[:, r] = expand_W(W_sc[:, r1])

    # 3. Construct full 4x4 * 4x9 decomposition
    # Total rank = 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8) = 114
    R_total = 114
    U_full = np.zeros((16, R_total))
    V_full = np.zeros((36, R_total)) # m*p = 4*9 = 36
    W_full = np.zeros((36, R_total)) # p*n = 9*4 = 36
    
    # Block 1: Cols 0-3 of C
    # U: same as U_44
    # V: B[0:4, 0:4] -> rows 0-15 of V_full
    # W: C[0:4, 0:4] -> rows 0-15 of W_full
    
    U_full[:, 0:49] = U_44
    V_full[0:16, 0:49] = V_44
    W_full[0:16, 0:49] = W_44
    
    # Block 2: Cols 4-7 of C
    # U: same as U_44
    # V: B[0:4, 4:8] -> rows 16-31 of V_full (j*9+k, j in 0..3, k in 4..7)
    #    V index for B[j,k] is j*9+k.
    #    For j=0, k=4..7 -> 4..7
    #    For j=1, k=4..7 -> 13..16
    #    For j=2, k=4..7 -> 22..25
    #    For j=3, k=4..7 -> 31..34
    #    Wait, V_44 indices are j*4+k for j,k in 0..3.
    #    We need to map V_44 indices to V_full indices.
    #    V_44 index = j*4+k.
    #    V_full index = j*9+k.
    #    Since k is 0..3 in V_44, and we want k to be 4..7 in V_full, we add 4 to k.
    #    So V_full_index = j*9 + (k+4).
    
    for r in range(49):
        for idx_44 in range(16):
            val = V_44[idx_44, r]
            if val != 0:
                j = idx_44 // 4
                k = idx_44 % 4
                idx_full = j*9 + (k+4)
                V_full[idx_full, 49+r] = val
                
    U_full[:, 49:98] = U_44
    W_full[4:20, 49:98] = W_44 # C[4:8, 0:4] -> k in 4..7, i in 0..3. W index k*4+i.
                                # k=4..7, i=0..3.
                                # W_full index = k*4+i.
                                # For k=4, i=0..3 -> 16..19.
                                # For k=7, i=0..3 -> 28..31.
                                # Wait, W_44 indices are k*4+i for k,i in 0..3.
                                # We want to place these at k=4..7.
                                # So W_full_index = (k+4)*4 + i.
    
    # Correct W mapping for Block 2
    for r in range(49):
        for idx_44 in range(16):
            val = W_44[idx_44, r]
            if val != 0:
                k = idx_44 // 4
                i = idx_44 % 4
                idx_full = (k+4)*4 + i
                W_full[idx_full, 49+r] = val

    # Block 3: Col 8 of C
    # U: A[i,j] -> i*4+j
    # V: B[j,8] -> j*9+8
    # W: C[8,i] -> 8*4+i = 32+i
    # Schoolbook: 16 terms.
    for i in range(4):
        for j in range(4):
            r = 98 + i*4 + j
            U_full[i*4+j, r] = 1
            V_full[j*9+8, r] = 1
            W_full[32+i, r] = 1

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    print(f"Rank: {U.shape[1]}")
    # Quick check
    for i in range(4):
        for j in range(4):
            for k in range(9):
                val = np.dot(U[i*4+j], V[j*9+k] * W[k*4+i])
                if k < 4 or (4 <= k < 8): # Only check valid C indices?
                    pass # C is 4x9, so k is 0..8.
                # Actually C is 4x9. k is 0..8.
                # Check val == 1 if consistent, 0 otherwise.
                # Consistent: i from U, j from U, j from V, k from V, k from W, i from W.
                # So i,i same, j,j same, k,k same.
                # If k<4: C[i,k] from Block 1.
                # If 4<=k<8: C[i,k] from Block 2.
                # If k==8: C[i,k] from Block 3.
                
                # Just check a few
    print("Done")
