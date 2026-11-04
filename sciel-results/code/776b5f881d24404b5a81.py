import numpy as np

def construct():
    n, m, p = 4, 4, 7
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: M_{4,4,4} using Strassen
    # Rank 49
    # U: 16x49, V: 16x49, W: 16x49
    
    # Strassen for 4x4x4
    # 7 multiplications of 2x2x2
    # Each 2x2x2 has rank 7
    # Total rank 49
    
    # Helper to get Strassen 2x2x2 tensors
    def get_strassen_222():
        # Returns U, V, W of shape 4x7 for 2x2x2
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # Wait, c21 = M2+M4, c22 = M1-M2+M3+M6
        # Let's re-verify Strassen formulas
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # M1: c11 += 1, c22 += 1
        # M2: c21 += 1, c22 -= 1
        # M3: c12 += 1, c22 += 1
        # M4: c11 += 1, c21 += 1
        # M5: c11 -= 1, c12 += 1
        # M6: c22 += 1
        # M7: c11 += 1
        
        # Correcting W assignments
        W[:, 0] = [1, 0, 0, 1] # M1
        W[:, 1] = [0, 0, 1, -1] # M2
        W[:, 2] = [0, 1, 0, 1] # M3
        W[:, 3] = [1, 0, 1, 0] # M4
        W[:, 4] = [-1, 1, 0, 0] # M5
        W[:, 5] = [0, 0, 0, 1] # M6
        W[:, 6] = [1, 0, 0, 0] # M7
        
        # U assignments
        # M1: a11, a22
        U[:, 0] = [1, 0, 0, 1]
        # M2: a21, a22
        U[:, 1] = [0, 0, 1, 1]
        # M3: a11
        U[:, 2] = [1, 0, 0, 0]
        # M4: a22
        U[:, 3] = [0, 0, 0, 1]
        # M5: a11, a12
        U[:, 4] = [1, 1, 0, 0]
        # M6: a21, a11
        U[:, 5] = [-1, 0, 1, 0]
        # M7: a12, a22
        U[:, 6] = [0, 1, 0, -1]
        
        # V assignments
        # M1: b11, b22
        V[:, 0] = [1, 0, 0, 1]
        # M2: b11
        V[:, 1] = [1, 0, 0, 0]
        # M3: b12, b22
        V[:, 2] = [0, 1, 0, -1]
        # M4: b21, b11
        V[:, 3] = [-1, 0, 1, 0]
        # M5: b22
        V[:, 4] = [0, 0, 0, 1]
        # M6: b11, b12
        V[:, 5] = [1, 1, 0, 0]
        # M7: b21, b22
        V[:, 6] = [0, 0, 1, 1]
        
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()
    
    # Now build 4x4x4 tensor
    # 7 blocks of 2x2x2
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Block indices
    # A blocks: 00 (0-3), 01 (4-7), 10 (8-11), 11 (12-15)
    # B blocks: 00 (0-3), 01 (4-7), 10 (8-11), 11 (12-15)
    # C blocks: 00 (0-3), 01 (4-7), 10 (8-11), 11 (12-15)
    
    # Strassen combinations
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_tensor_block(U_target, V_target, W_target, 
                         U_src, V_src, W_src, 
                         u_offset, v_offset, w_offset, 
                         u_sign=1, v_sign=1, w_sign=1,
                         col_offset=0):
        # Add scaled/shifted source tensor to target
        # u_offset, v_offset, w_offset are row offsets
        # col_offset is column offset
        # u_sign, v_sign, w_sign are multipliers
        
        for c in range(7):
            # U
            for r in range(4):
                if U_src[r, c] != 0:
                    U_target[u_offset + r, col_offset + c] += u_sign * U_src[r, c]
            # V
            for r in range(4):
                if V_src[r, c] != 0:
                    V_target[v_offset + r, col_offset + c] += v_sign * V_src[r, c]
            # W
            for r in range(4):
                if W_src[r, c] != 0:
                    W_target[w_offset + r, col_offset + c] += w_sign * W_src[r, c]

    # M1
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 0, 0, 1, 1, 1, 0)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 12, 12, 1, 1, 1, 0)
    
    # M2
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 8, 0, 8, 1, 1, 1, 7)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 0, 12, 1, 1, 1, 7)
    
    # M3
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 4, 4, 1, 1, 1, 14)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 12, 12, 1, -1, 1, 14)
    
    # M4
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 8, 8, 1, 1, 1, 21)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 0, 0, 1, -1, 1, 21)
    
    # M5
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 12, 12, 1, 1, 1, 28)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 4, 12, 12, 1, 1, 1, 28)
    
    # M6
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 8, 0, 8, 1, 1, 1, 35)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 0, 0, -1, 1, 1, 35)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 8, 4, 8, 1, 1, 1, 35)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 0, 4, 4, -1, 1, 1, 35)
    
    # M7
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 4, 8, 8, 1, 1, 1, 42)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 8, 8, -1, 1, 1, 42)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 4, 12, 12, 1, 1, 1, 42)
    add_tensor_block(U_444, V_444, W_444, U_222, V_222, W_222, 12, 12, 12, -1, 1, 1, 42)

    U_list.append(U_444)
    V_list.append(V_444)
    W_list.append(W_444)
    total_rank += 49

    # Part 2: M_{4,4,3}
    # Rank 48
    # U: 16x48, V: 12x48, W: 12x48
    
    # M_{2,2,3} rank 12
    # M_{2,2,1} rank 4
    # Tensor product
    
    # Get M_{2,2,1}
    # Schoolbook: 4 terms
    U_221 = np.zeros((4, 4), dtype=int)
    V_221 = np.zeros((4, 4), dtype=int)
    W_221 = np.zeros((4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            col = i * 2 + j
            U_221[i, col] = 1
            V_221[j, col] = 1
            W_221[i, col] = 1 # c_i = sum a_ij b_j. Here c is 2x1. c_i.
            # Wait, M_{2,2,1} output is 2x1.
            # W rows: c_0, c_1.
            # U rows: a_00, a_01, a_10, a_11.
            # V rows: b_0, b_1.
            
            # Correction:
            # c_0 = a_00 b_0 + a_01 b_1
            # c_1 = a_10 b_0 + a_11 b_1
            
            # Term 1: a_00 b_0 -> c_0
            # U[0]=1, V[0]=1, W[0]=1
            # Term 2: a_01 b_1 -> c_0
            # U[1]=1, V[1]=1, W[0]=1
            # Term 3: a_10 b_0 -> c_1
            # U[2]=1, V[0]=1, W[1]=1
            # Term 4: a_11 b_1 -> c_1
            # U[3]=1, V[1]=1, W[1]=1
            
    U_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # M_{2,2,3} is 3 copies of M_{2,2,1}
    U_223 = np.zeros((4, 12), dtype=int)
    V_223 = np.zeros((2, 12), dtype=int)
    W_223 = np.zeros((2, 12), dtype=int)
    for k in range(3):
        col_offset = k * 4
        U_223[:, col_offset:col_offset+4] = U_221
        V_223[:, col_offset:col_offset+4] = V_221
        W_223[:, col_offset:col_offset+4] = W_221

    # Tensor Product M_{4,4,3} = M_{2,2,3} (p=3) x M_{2,2,1} (n=2,m=2) ?
    # No, dimensions:
    # M_{4,4,3}: n=4, m=4, p=3.
    # Split n=2*2, m=2*2, p=3*1.
    # Factor 1: M_{2,2,3} (n=2, m=2, p=3). Rank 12.
    # Factor 2: M_{2,2,1} (n=2, m=2, p=1). Rank 4.
    # Total Rank 48.
    
    # U_total = U_1 * U_2 (Kronecker product of rows)
    # V_total = V_1 * V_2
    # W_total = W_1 * W_2
    
    # U_1 is 4x12 (n1*m1 = 2*2=4)
    # U_2 is 4x4 (n2*m2 = 2*2=4)
    # U_total is 16x48
    
    # We need to construct the Kronecker product of the tensors.
    # The columns of the total tensor correspond to pairs of columns from the factors.
    # col_total = col1 * rank2 + col2
    
    U_443 = np.zeros((16, 48), dtype=int)
    V_443 = np.zeros((12, 48), dtype=int) # m1*p1=2*3=6, m2*p2=2*1=2. Total 12?
    # Wait. V dimension is m*p.
    # M_{2,2,3}: m=2, p=3. V size 6.
    # M_{2,2,1}: m=2, p=1. V size 2.
    # Total V size 6*2=12. Correct.
    
    W_443 = np.zeros((12, 48), dtype=int)
    # W dimension is p*n.
    # M_{2,2,3}: p=3, n=2. W size 6.
    # M_{2,2,1}: p=1, n=2. W size 2.
    # Total W size 6*2=12. Correct.
    
    for c1 in range(12):
        for c2 in range(4):
            c_tot = c1 * 4 + c2
            # U
            # Row index in U_total: r1 * 4 + r2
            # r1 in 0..3, r2 in 0..3
            for r1 in range(4):
                if U_223[r1, c1] == 0: continue
                for r2 in range(4):
                    if U_221[r2, c2] == 0: continue
                    U_443[r1 * 4 + r2, c_tot] = U_223[r1, c1] * U_221[r2, c2]
            
            # V
            # Row index in V_total: r1 * 2 + r2
            # r1 in 0..5, r2 in 0..1
            for r1 in range(6):
                if V_223[r1, c1] == 0: continue
                for r2 in range(2):
                    if V_221[r2, c2] == 0: continue
                    V_443[r1 * 2 + r2, c_tot] = V_223[r1, c1] * V_221[r2, c2]
            
            # W
            # Row index in W_total: r1 * 2 + r2
            # r1 in 0..5, r2 in 0..1
            for r1 in range(6):
                if W_223[r1, c1] == 0: continue
                for r2 in range(2):
                    if W_221[r2, c2] == 0: continue
                    W_443[r1 * 2 + r2, c_tot] = W_223[r1, c1] * W_221[r2, c2]
    
    # Map V and W indices to global positions
    # V global: m=4, p=7. Total 28.
    # M_{4,4,3} corresponds to columns 4..6 of B.
    # Local V rows: r1 (0..5) from M_{2,2,3}, r2 (0..1) from M_{2,2,1}.
    # M_{2,2,3} V rows: m=2, p=3. Indices j1 (0..1), k1 (0..2).
    # M_{2,2,1} V rows: m=2, p=1. Indices j2 (0..1), k2 (0..0).
    # Global j = j1 * 2 + j2.
    # Global k = 4 + k1 * 1 + k2. (Since p1=3, p2=1, offset 4).
    # Local V index = r1 * 2 + r2.
    # r1 = j1 * 3 + k1.
    # r2 = j2 * 1 + k2 = j2.
    
    V_443_global = np.zeros((28, 48), dtype=int)
    for c in range(48):
        for r_local in range(12):
            if V_443[r_local, c] == 0: continue
            r1 = r_local // 2
            r2 = r_local % 2
            j1 = r1 // 3
            k1 = r1 % 3
            j2 = r2
            k2 = 0
            j_global = j1 * 2 + j2
            k_global = 4 + k1 + k2
            r_global = j_global * 7 + k_global
            V_443_global[r_global, c] = V_443[r_local, c]
    
    # W global: p=7, n=4. Total 28.
    # M_{4,4,3} corresponds to columns 4..6 of C.
    # Local W rows: r1 (0..5) from M_{2,2,3}, r2 (0..1) from M_{2,2,1}.
    # M_{2,2,3} W rows: p=3, n=2. Indices k1 (0..2), i1 (0..1).
    # M_{2,2,1} W rows: p=1, n=2. Indices k2 (0..0), i2 (0..1).
    # Global k = 4 + k1 * 1 + k2.
    # Global i = i1 * 2 + i2.
    # Local W index = r1 * 2 + r2.
    # r1 = k1 * 2 + i1.
    # r2 = k2 * 2 + i2 = i2.
    
    W_443_global = np.zeros((28, 48), dtype=int)
    for c in range(48):
        for r_local in range(12):
            if W_443[r_local, c] == 0: continue
            r1 = r_local // 2
            r2 = r_local % 2
            k1 = r1 // 2
            i1 = r1 % 2
            i2 = r2
            k2 = 0
            k_global = 4 + k1 + k2
            i_global = i1 * 2 + i2
            r_global = k_global * 4 + i_global
            W_443_global[r_global, c] = W_443[r_local, c]
    
    # U_443 is already in global form (n=4, m=4).
    # Local U rows: r1 (0..3) from M_{2,2,3}, r2 (0..3) from M_{2,2,1}.
    # r1 = i1 * 2 + j1.
    # r2 = i2 * 2 + j2.
    # Global i = i1 * 2 + i2.
    # Global j = j1 * 2 + j2.
    # This mapping is identity if we order blocks correctly.
    # M_{2,2,3} U: i1, j1.
    # M_{2,2,1} U: i2, j2.
    # Global i = i1*2 + i2.
    # Global j = j1*2 + j2.
    # Local index = r1*4 + r2 = (i1*2+j1)*4 + (i2*2+j2).
    # Global index = i_global*m + j_global = (i1*2+i2)*4 + (j1*2+j2).
    # They are not the same. We need to permute U_443 rows.
    
    U_443_global = np.zeros((16, 48), dtype=int)
    for c in range(48):
        for r_local in range(16):
            if U_443[r_local, c] == 0: continue
            r1 = r_local // 4
            r2 = r_local % 4
            i1 = r1 // 2
            j1 = r1 % 2
            i2 = r2 // 2
            j2 = r2 % 2
            i_global = i1 * 2 + i2
            j_global = j1 * 2 + j2
            r_global = i_global * 4 + j_global
            U_443_global[r_global, c] = U_443[r_local, c]
    
    U_list.append(U_443_global)
    V_list.append(V_443_global)
    W_list.append(W_443_global)
    total_rank += 48

    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
