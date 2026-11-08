import numpy as np

def construct():
    n, m, p = 3, 4, 8
    # Rank 32 decomposition
    # C0: 1x8 from 1x4 (Rank 4)
    # C1: 2x8 from 2x4 (Rank 28)
    
    U = []
    V = []
    W = []
    
    # Part 1: 1x4x8 (Rank 4)
    # A0 is 1x4, B is 4x8
    # Decompose 1x4x8 as (1x2x4) x (1x2x2) ? No, 1x4x8 is just rank 4.
    # U shape (1*4, 4), V shape (4*8, 4), W shape (8*1, 4)
    # Actually, 1x4x8 tensor:
    # T_{0,j,k} = 1 if j,k independent?
    # C_k = sum_j A_j B_{j,k}
    # This is rank 4.
    # U_col r: A_r = 1, others 0.
    # V_col r: B_{r,k} = 1 for all k? No.
    # V_{j,k} for fixed r is 1 if j=r, else 0.
    # W_{k,0} for fixed r is 1.
    
    for r in range(4):
        u = [0]*4
        u[r] = 1
        v = [0]*32
        for k in range(8):
            v[r*8+k] = 1
        w = [0]*8
        w[k*1+0] = 1 # W index k*n + i = k*1 + 0 = k
        # Wait, W index is k*n + i. i=0. So index k.
        # But W shape is (p*n, rank) = (8, 4).
        # So W[k, r] = 1.
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Part 2: 2x4x8 (Rank 28)
    # A1 is 2x4, B is 4x8
    # Split into two 2x4x4 blocks.
    # Block 1: 2x4x4 (Rank 14)
    # Block 2: 2x4x4 (Rank 14)
    
    # Strassen 2x2x2 (Rank 7)
    # Factors for 2x2x2:
    # U_s (4x7), V_s (4x7), W_s (4x7)
    # Standard Strassen factors (0/1/-1)
    U_s = [
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0]
    ]
    # This is schoolbook. Strassen is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U_s (4x7):
    # A11=1, A12=0, A21=0, A22=0 -> indices 0,1,2,3
    U_strassen = [
        [1,0,1,0,1,0,0], # A11
        [0,1,0,1,0,1,0], # A12
        [0,0,1,0,0,1,0], # A21
        [1,1,0,1,1,0,1]  # A22
    ]
    V_strassen = [
        [1,1,0,0,0,1,0], # B11
        [0,0,1,1,0,0,1], # B12
        [1,0,1,0,1,0,1], # B21
        [1,0,0,1,1,0,0]  # B22
    ]
    W_strassen = [
        [1,0,0,0,0,0,0], # C11
        [0,1,0,0,0,0,0], # C12
        [0,0,1,0,0,0,0], # C21
        [0,0,0,1,0,0,0]  # C22
    ]
    # Note: W indices are k*n+i. For 2x2, n=2.
    # C11 (i=0,k=0) -> 0. C12 (i=0,k=1) -> 1.
    # C21 (i=1,k=0) -> 2. C22 (i=1,k=1) -> 3.
    # So W_strassen is identity 4x4? No, it's 4x7.
    # C11 = M1 - M4 + M6 - M7?
    # Standard Strassen W:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    W_strassen = [
        [1, -1, 0, 1, 0, 0, 0], # C11
        [0, 0, 1, 0, 1, 0, 0], # C12
        [0, 1, 0, 1, 0, 0, 0], # C21
        [1, 0, -1, 0, 1, 0, 1] # C22
    ]
    
    # 1x2x2 (Rank 2)
    U_122 = [
        [1, 1], # A1
        [1, 1]  # A2
    ]
    V_122 = [
        [1, 0], # B11
        [0, 1], # B12
        [1, 0], # B21
        [0, 1]  # B22
    ]
    W_122 = [
        [1, 0], # C1
        [0, 1]  # C2
    ]
    
    # Combine 2x2x2 and 1x2x2 to get 2x4x4
    # U (8x14), V (32x14), W (16x14)
    # Wait, 2x4x4: U (8x14), V (16x14), W (8x14).
    # U indices: i*m+j. m=4. i in 0..1, j in 0..3.
    # V indices: j*p+k. p=4. j in 0..3, k in 0..3.
    # W indices: k*n+i. n=2. k in 0..3, i in 0..1.
    
    U_244 = []
    V_244 = []
    W_244 = []
    
    for r_s in range(7):
        for r_122 in range(2):
            u = [0]*8
            v = [0]*16
            w = [0]*8
            
            # Map U_strassen (4x7) to U (8x14)
            # U_strassen index: i*2+j (2x2).
            # U_244 index: i*4+j (2x4).
            # j = j_s * 2 + j_122?
            # m = 2*2. j_s in 0..1, j_122 in 0..1.
            # j = j_s * 2 + j_122.
            
            # U_strassen[i_s, r_s] * U_122[j_122, r_122]
            # i = i_s. j = j_s*2 + j_122.
            
            for i_s in range(2):
                for j_s in range(2):
                    for j_122 in range(2):
                        val = U_strassen[i_s*2+j_s, r_s] * U_122[j_122, r_122]
                        idx_u = i_s*4 + (j_s*2 + j_122)
                        u[idx_u] += val
                        
            for j_s in range(2):
                for k_s in range(2):
                    for j_122 in range(2):
                        for k_122 in range(2):
                            # V_strassen index: j_s*2+k_s.
                            # V_122 index: j_122*2+k_122.
                            # V_244 index: j*4+k.
                            # j = j_s*2+j_122. k = k_s*2+k_122.
                            val = V_strassen[j_s*2+k_s, r_s] * V_122[j_122*2+k_122, r_122]
                            idx_v = (j_s*2+j_122)*4 + (k_s*2+k_122)
                            v[idx_v] += val
                            
            for k_s in range(2):
                for i_s in range(2):
                    for k_122 in range(2):
                        # W_strassen index: k_s*2+i_s.
                        # W_122 index: k_122.
                        # W_244 index: k*2+i.
                        # k = k_s*2+k_122. i = i_s.
                        val = W_strassen[k_s*2+i_s, r_s] * W_122[k_122, r_122]
                        idx_w = (k_s*2+k_122)*2 + i_s
                        w[idx_w] += val
                        
            U_244.append(u)
            V_244.append(v)
            W_244.append(w)
            
    # Now combine two 2x4x4 blocks for 2x4x8
    # Block 1: k in 0..3. Block 2: k in 4..7.
    # U is same.
    # V: V_block1 (16x14), V_block2 (16x14).
    # W: W_block1 (8x14), W_block2 (8x14).
    
    U_248 = []
    V_248 = []
    W_248 = []
    
    for r in range(14):
        u = U_244[r]
        v = [0]*32
        w = [0]*16
        
        # Block 1
        for idx in range(16):
            v[idx] = V_244[r][idx]
            w[idx] = W_244[r][idx]
            
        # Block 2
        for idx in range(16):
            v[16+idx] = V_244[r][idx]
            w[8+idx] = W_244[r][idx]
            
        U_248.append(u)
        V_248.append(v)
        W_248.append(w)
        
    # Combine Part 1 and Part 2
    # Part 1: 1x4x8 (Rank 4). U (4x4), V (32x4), W (8x4).
    # Part 2: 2x4x8 (Rank 28). U (8x28), V (32x28), W (16x28).
    # Total U: 12x32.
    # Total V: 32x32.
    # Total W: 24x32.
    
    U_final = []
    V_final = []
    W_final = []
    
    # Part 1
    for r in range(4):
        u = [0]*12
        u[r] = 1 # A0_r
        # V
        v = [0]*32
        for k in range(8):
            v[r*8+k] = 1
        # W
        w = [0]*24
        w[k*3+0] = 1 # i=0, k in 0..7. Index k*3+0.
        # Wait, W index is k*n+i. n=3.
        # For i=0, index is k*3+0.
        
        U_final.append(u)
        V_final.append(v)
        W_final.append(w)
        
    # Part 2
    # A1 is rows 1,2 of A.
    # U indices for A1: i=1,2. j=0..3.
    # U_248 indices: i*4+j. i in 0..1.
    # Map i_248=0 -> i_final=1. i_248=1 -> i_final=2.
    for r in range(28):
        u = [0]*12
        for i_248 in range(2):
            for j in range(4):
                idx_248 = i_248*4+j
                idx_final = (i_248+1)*4+j
                u[idx_final] = U_248[r][idx_248]
                
        v = V_248[r]
        
        w = [0]*24
        for i_248 in range(2):
            for k in range(4):
                idx_248 = k*2+i_248
                idx_final = k*3 + (i_248+1)
                w[idx_final] = W_248[r][idx_248]
                
        U_final.append(u)
        V_final.append(v)
        W_final.append(w)
        
    return {"U": U_final, "V": V_final, "W": W_final}
