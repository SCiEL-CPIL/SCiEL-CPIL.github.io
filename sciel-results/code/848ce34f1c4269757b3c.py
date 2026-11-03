import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline to be safe
    
    n, m, p = DIMENSIONS
    
    # Partition indices
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=7 -> [2, 2, 2, 1]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2, 1]
    
    # Precompute offsets
    n_offsets = []
    curr = 0
    for size in n_parts:
        n_offsets.append(curr)
        curr += size
    m_offsets = []
    curr = 0
    for size in m_parts:
        m_offsets.append(curr)
        curr += size
    p_offsets = []
    curr = 0
    for size in p_parts:
        p_offsets.append(curr)
        curr += size
        
    U_list = []
    V_list = []
    W_list = []
    
    for i_idx, n_size in enumerate(n_parts):
        for j_idx, m_size in enumerate(m_parts):
            for k_idx, p_size in enumerate(p_parts):
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Strassen 2x2x2
                    # Local indices i, j, k in 0..1
                    # Global indices:
                    # i_global = n_offsets[i_idx] + i
                    # j_global = m_offsets[j_idx] + j
                    # k_global = p_offsets[k_idx] + k
                    
                    # Strassen factors
                    # M1: A00(B00+B11) -> U00, V00+V11, W00? No.
                    # We need U, V, W such that sum U_ij V_jk W_ki = 1.
                    
                    # Strassen decomposition of 2x2x2 tensor:
                    # 7 terms.
                    # Term 1: A00 * (B00+B11)
                    # Term 2: (A10+A11) * B00
                    # Term 3: A01 * (B10-B00)
                    # Term 4: A11 * (B11-B01)
                    # Term 5: (A00+A01) * B11
                    # Term 6: (A10-A00) * (B00+B01)
                    # Term 7: (A01-A11) * (B10+B11)
                    
                    # The factors U, V, W for each term r are:
                    # U_r[i, j], V_r[j, k], W_r[k, i]
                    
                    # Term 1:
                    # U1[0,0]=1, others 0.
                    # V1[0,0]=1, V1[1,1]=1.
                    # W1[0,0]=1, W1[1,1]=1? No, W is for output C.
                    # C00 gets M1. C11 gets M1?
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # So W1[0,0] = 1 (for C00), W1[1,1] = 1 (for C11)?
                    # Wait, W is indexed by (k, i).
                    # W1[k, i] is the coefficient for C[i, k].
                    # M1 contributes to C00 and C11.
                    # So W1[0,0] = 1, W1[1,1] = 1.
                    
                    # Let's build the 7 terms.
                    
                    # Term 1
                    U1 = [[0,0],[0,0]]; U1[0][0] = 1
                    V1 = [[0,0],[0,0]]; V1[0][0] = 1; V1[1][1] = 1
                    W1 = [[0,0],[0,0]]; W1[0][0] = 1; W1[1][1] = 1
                    
                    # Term 2
                    U2 = [[0,0],[0,0]]; U2[1][0] = 1; U2[1][1] = 1
                    V2 = [[0,0],[0,0]]; V2[0][0] = 1
                    W2 = [[0,0],[0,0]]; W2[0][1] = 1; W2[1][1] = -1 # C10, C11?
                    # C10 = M2 + M4. C11 = M1 - M2 + ...
                    # So W2[0,1] = 1 (C10), W2[1,1] = -1 (C11).
                    
                    # Term 3
                    U3 = [[0,0],[0,0]]; U3[0][1] = 1
                    V3 = [[0,0],[0,0]]; V3[1][0] = 1; V3[0][0] = -1
                    W3 = [[0,0],[0,0]]; W3[0][0] = 1; W3[1][1] = -1 # C01, C11?
                    # C01 = M3 + M5. C11 = ... + M3 ...
                    # So W3[0,0] = 1 (C01), W3[1,1] = 1 (C11).
                    
                    # Term 4
                    U4 = [[0,0],[0,0]]; U4[1][1] = 1
                    V4 = [[0,0],[0,0]]; V4[1][1] = 1; V4[1][0] = -1
                    W4 = [[0,0],[0,0]]; W4[0][0] = 1; W4[1][0] = 1 # C00, C10?
                    # C00 = ... + M4 ... C10 = M2 + M4.
                    # So W4[0,0] = 1 (C00), W4[0,1] = 1 (C10).
                    
                    # Term 5
                    U5 = [[0,0],[0,0]]; U5[0][0] = 1; U5[0][1] = 1
                    V5 = [[0,0],[0,0]]; V5[1][1] = 1
                    W5 = [[0,0],[0,0]]; W5[0][0] = -1; W5[0][1] = 1 # C00, C01?
                    # C00 = ... - M5 ... C01 = M3 + M5.
                    # So W5[0,0] = -1 (C00), W5[0][1] = 1 (C01).
                    
                    # Term 6
                    U6 = [[0,0],[0,0]]; U6[1][0] = 1; U6[0][0] = -1
                    V6 = [[0,0],[0,0]]; V6[0][0] = 1; V6[0][1] = 1
                    W6 = [[0,0],[0,0]]; W6[1][1] = 1 # C11?
                    # C11 = ... + M6.
                    # So W6[1][1] = 1 (C11).
                    
                    # Term 7
                    U7 = [[0,0],[0,0]]; U7[0][1] = 1; U7[1][1] = -1
                    V7 = [[0,0],[0,0]]; V7[1][0] = 1; V7[1][1] = 1
                    W7 = [[0,0],[0,0]]; W7[0][0] = 1 # C00?
                    # C00 = ... + M7.
                    # So W7[0][0] = 1 (C00).
                    
                    terms = [
                        (U1, V1, W1),
                        (U2, V2, W2),
                        (U3, V3, W3),
                        (U4, V4, W4),
                        (U5, V5, W5),
                        (U6, V6, W6),
                        (U7, V7, W7)
                    ]
                    
                    for U_t, V_t, W_t in terms:
                        # Flatten and add to lists
                        U_flat = [0] * (n*m)
                        V_flat = [0] * (m*p)
                        W_flat = [0] * (p*n)
                        
                        for li in range(2):
                            for lj in range(2):
                                gi = n_offsets[i_idx] + li
                                gj = m_offsets[j_idx] + lj
                                U_flat[gi*m + gj] = U_t[li][lj]
                                
                        for lj in range(2):
                            for lk in range(2):
                                gj = m_offsets[j_idx] + lj
                                gk = p_offsets[k_idx] + lk
                                V_flat[gj*p + gk] = V_t[lj][lk]
                                
                        for lk in range(2):
                            for li in range(2):
                                gk = p_offsets[k_idx] + lk
                                gi = n_offsets[i_idx] + li
                                W_flat[gk*n + gi] = W_t[lk][li]
                                
                        U_list.append(U_flat)
                        V_list.append(V_flat)
                        W_list.append(W_flat)
                        
                else:
                    # Schoolbook for this block
                    for li in range(n_size):
                        for lj in range(m_size):
                            for lk in range(p_size):
                                U_flat = [0] * (n*m)
                                V_flat = [0] * (m*p)
                                W_flat = [0] * (p*n)
                                
                                gi = n_offsets[i_idx] + li
                                gj = m_offsets[j_idx] + lj
                                gk = p_offsets[k_idx] + lk
                                
                                U_flat[gi*m + gj] = 1
                                V_flat[gj*p + gk] = 1
                                W_flat[gk*n + gi] = 1
                                
                                U_list.append(U_flat)
                                V_list.append(V_flat)
                                W_list.append(W_flat)
                                
                                if time.time() > deadline:
                                    break
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we hit deadline, we might have incomplete lists. 
    # But the loop structure is small (24 blocks), so it should be fast.
    
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list}

