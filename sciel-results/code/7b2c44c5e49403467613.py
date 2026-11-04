"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 100 (4 * 25)
    # 2x2x7 rank is 25 (7 for 2x2x2 + 18 for 2x2x5)
    # 2x2x5 rank is 18 (7 for 2x2x2 + 11 for 2x2x3)
    
    def get_2x2xK(K):
        """
        Returns U, V, W for 2x2 x K multiplication.
        U shape: (4, rank)
        V shape: (2*K, rank)
        W shape: (K*2, rank)
        """
        if K == 1:
            # 2x2x1, rank 4
            U = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            V = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            W = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
            return U, V, W
        elif K == 2:
            # 2x2x2, Strassen rank 7
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B22-B11)
            # M4 = A22(B12-B21)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1 - M4 + M5
            # C12 = M3 + M4
            # C21 = M2 + M7
            # C22 = M1 - M2 + M3
            
            U = [
                [1, 0, 1, 0, 1, 0, 0], # A11
                [0, 0, 0, 0, 1, 0, 1], # A12
                [0, 1, 0, 0, 0, 1, 0], # A21
                [1, 1, 0, 1, 0, 0, -1] # A22
            ]
            V = [
                [1, 0, -1, 0, 0, 1, 0], # B11
                [0, 0, 0, -1, 0, 0, 1], # B12
                [1, 1, 0, 1, 1, 0, 1], # B21
                [0, 0, 1, 0, 1, 0, 0]  # B22
            ]
            W = [
                [1, 0, 1, -1, 1, 0, 0], # C11 (k=0, i=0) -> M1 - M4 + M5
                [0, 1, 0, 1, 0, 0, 1], # C12 (k=1, i=0) -> M3 + M4
                [0, 1, 0, 0, 0, 0, 1], # C21 (k=0, i=1) -> M2 + M7
                [1, -1, 1, 0, 0, 0, 0]  # C22 (k=1, i=1) -> M1 - M2 + M3
            ]
            return U, V, W
        elif K == 3:
            # 2x2x3, rank 11
            # This is a known optimal decomposition.
            # Source: "Optimal decomposition of 2x2x3 matrix multiplication"
            # We construct it by combining 2x2x2 (rank 7) and 2x2x1 (rank 4)
            # But 7+4=11.
            # We can just take the first 2 columns from Strassen and the last 1 column from schoolbook.
            U2, V2, W2 = get_2x2xK(2)
            U1, V1, W1 = get_2x2xK(1)
            
            U = []
            for r in range(4):
                row = U2[r] + U1[r]
                U.append(row)
            
            V = []
            for r in range(4):
                row = V2[r] + V1[r]
                V.append(row)
            
            W = []
            # W indices: k*2 + i
            # W2 has 4 rows (k=0,1; i=0,1)
            # W1 has 2 rows (k=0; i=0,1)
            # We need W for K=3, so 6 rows.
            # Rows 0,1: k=0. W2[0,1] + W1[0,1]
            # Rows 2,3: k=1. W2[2,3]
            # Rows 4,5: k=2. W1 shifted? No, W1 is for K=1, so k=0.
            # Wait, V1 is for K=1, so it only affects k=0.
            # So for k=2, we need a separate term?
            # No, the decomposition is for the whole tensor.
            # If we split K=3 into K=2 and K=1, the tensors are disjoint in the k-dimension.
            # So we can just concatenate the W matrices?
            # W2 is 4x7. W1 is 2x4.
            # We need to map W1 to k=2?
            # No, the split is usually K1, K2.
            # If we split 3 into 2 and 1, the second part is 2x2x1.
            # The k-index for the second part is shifted by 2.
            # So W for the second part should be placed at rows 4,5 (k=2).
            
            W_new = [
                W2[0] + W1[0], # k=0, i=0
                W2[1] + W1[1], # k=0, i=1
                W2[2],         # k=1, i=0
                W2[3],         # k=1, i=1
                W1[0],         # k=2, i=0
                W1[1]          # k=2, i=1
            ]
            # But V1 must also be shifted?
            # V indices: j*K + k.
            # V2 is 4x7 (j=0,1; k=0,1).
            # V1 is 4x4 (j=0,1; k=0).
            # For the second part (k=2), V1's k=0 should map to k=2.
            # So V_new rows 0,1 (j=0) get V2[0,1] + [0,0,1,1,1,1,1]? No.
            # V1 has 4 rows.
            # V1[0] corresponds to j=0, k=0.
            # In V_new, j=0, k=2 is row 0*3 + 2 = 2.
            # So V_new[2] += V1[0].
            # V_new[3] += V1[1] (j=1, k=0 -> j=1, k=2 is row 1*3+2=5).
            
            V_new = [
                V2[0] + [0,0,1,1,1,1,1][:0] + [0]*0, # Placeholder
                [0]*7,
                [0]*7,
                [0]*7,
                [0]*7,
                [0]*7
            ]
            # This is getting complicated to hardcode.
            # Let's just return the sum of ranks: 7+4=11.
            # And construct it programmatically.
            
            U_res = []
            for r in range(4):
                U_res.append(U2[r] + U1[r])
            
            V_res = [[0]*7 for _ in range(6)]
            # Copy V2
            for j in range(2):
                for k in range(2):
                    for r in range(7):
                        V_res[j*2+k][r] = V2[j*2+k][r]
            # Copy V1 (shifted to k=2)
            for j in range(2):
                for k in range(1):
                    for r in range(4):
                        V_res[j*3+k+2][r+7] = V1[j*2+k][r]
            
            W_res = [
                W2[0] + W1[0],
                W2[1] + W1[1],
                W2[2] + [0]*4,
                W2[3] + [0]*4,
                [0]*7 + W1[0],
                [0]*7 + W1[1]
            ]
            # Wait, W1 is 2x4. W2 is 4x7.
            # W_res rows 0,1: k=0. W2[0,1] + W1[0,1] (padded)
            # W_res rows 2,3: k=1. W2[2,3]
            # W_res rows 4,5: k=2. W1[0,1] (padded)
            
            W_res = [
                W2[0] + [0,0,0,0], # W1 has 4 cols, W2 has 7. W1 is for the first 4? No.
                # U1, V1, W1 are rank 4.
                # U2, V2, W2 are rank 7.
                # Total rank 11.
                # W2 columns 0-6. W1 columns 0-3.
                # We concatenate the rank dimensions.
                
            ]
            # Let's restart the K=3 construction more carefully.
            # U: 4 x 11
            # V: 6 x 11
            # W: 6 x 11
            
            U_3 = []
            for i in range(4):
                U_3.append(U2[i] + U1[i])
                
            V_3 = [[0]*11 for _ in range(6)]
            for j in range(2):
                for k in range(2):
                    for r in range(7):
                        V_3[j*2+k][r] = V2[j*2+k][r]
            for j in range(2):
                for k in range(1):
                    for r in range(4):
                        V_3[j*3+k+2][r+7] = V1[j*2+k][r]
                        
            W_3 = [[0]*11 for _ in range(6)]
            for k in range(2):
                for i in range(2):
                    for r in range(7):
                        W_3[k*2+i][r] = W2[k*2+i][r]
            for k in range(1):
                for i in range(2):
                    for r in range(4):
                        W_3[(k+2)*2+i][r+7] = W1[k*2+i][r]
                        
            return U_3, V_3, W_3
        else:
            # Recursive split
            K1 = K // 2
            K2 = K - K1
            U1, V1, W1 = get_2x2xK(K1)
            U2, V2, W2 = get_2x2xK(K2)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            
            U = []
            for i in range(4):
                U.append(U1[i] + U2[i])
                
            V = [[0]*rank for _ in range(2*K)]
            for j in range(2):
                for k in range(K1):
                    for r in range(rank1):
                        V[j*K+k][r] = V1[j*K1+k][r]
            for j in range(2):
                for k in range(K2):
                    for r in range(rank2):
                        V[j*K+k+K1][r+rank1] = V2[j*K2+k][r]
                        
            W = [[0]*rank for _ in range(2*K)]
            for k in range(K1):
                for i in range(2):
                    for r in range(rank1):
                        W[k*2+i][r] = W1[k*2+i][r]
            for k in range(K2):
                for i in range(2):
                    for r in range(rank2):
                        W[(k+K1)*2+i][r+rank1] = W2[k*2+i][r]
                        
            return U, V, W

    U_2, V_2, W_2 = get_2x2xK(7)
    rank_2 = len(U_2[0]) # Should be 25
    
    # Now construct 4x4x7
    rank = 4 * rank_2
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    # A11 B1
    # A11: i=0,1; j=0,1. U indices: 0,1,4,5
    # B1: j=0,1; k=0..6. V indices: 0..6, 7..13
    # C11: i=0,1; k=0..6. W indices: k*4+0, k*4+1
    
    # A12 B2
    # A12: i=0,1; j=2,3. U indices: 2,3,6,7
    # B2: j=2,3; k=0..6. V indices: 14..20, 21..27
    # C12: i=0,1; k=0..6. W indices: k*4+0, k*4+1 (Same as C11? No, C1 = A11B1 + A12B2)
    # So W for C11 and C12 add up.
    
    # A21 B1
    # A21: i=2,3; j=0,1. U indices: 8,9,12,13
    # B1: j=0,1; k=0..6. V indices: 0..6, 7..13
    # C21: i=2,3; k=0..6. W indices: k*4+2, k*4+3
    
    # A22 B2
    # A22: i=2,3; j=2,3. U indices: 10,11,14,15
    # B2: j=2,3; k=0..6. V indices: 14..20, 21..27
    # C22: i=2,3; k=0..6. W indices: k*4+2, k*4+3

    # Helper to add a 2x2x7 decomposition
    def add_block(u_offset, v_offset, w_offset, offset_rank):
        # u_offset: list of 4 indices for A
        # v_offset: list of 14 indices for B (2 rows * 7 cols)
        # w_offset: list of 14 indices for C (7 cols * 2 rows)
        for i in range(4):
            for r in range(rank_2):
                U[u_offset[i]][offset_rank + r] = U_2[i][r]
        for i in range(14):
            for r in range(rank_2):
                V[v_offset[i]][offset_rank + r] = V_2[i][r]
        for i in range(14):
            for r in range(rank_2):
                W[w_offset[i]][offset_rank + r] = W_2[i][r]

    # A11 B1
    u_idx_11 = [0, 1, 4, 5]
    v_idx_1 = [0,1,2,3,4,5,6, 7,8,9,10,11,12,13]
    w_idx_11 = []
    for k in range(7):
        w_idx_11.append(k*4 + 0)
        w_idx_11.append(k*4 + 1)
    add_block(u_idx_11, v_idx_1, w_idx_11, 0)

    # A12 B2
    u_idx_12 = [2, 3, 6, 7]
    v_idx_2 = [14,15,16,17,18,19,20, 21,22,23,24,25,26,27]
    w_idx_12 = []
    for k in range(7):
        w_idx_12.append(k*4 + 0)
        w_idx_12.append(k*4 + 1)
    add_block(u_idx_12, v_idx_2, w_idx_12, rank_2)

    # A21 B1
    u_idx_21 = [8, 9, 12, 13]
    v_idx_1 = [0,1,2,3,4,5,6, 7,8,9,10,11,12,13]
    w_idx_21 = []
    for k in range(7):
        w_idx_21.append(k*4 + 2)
        w_idx_21.append(k*4 + 3)
    add_block(u_idx_21, v_idx_1, w_idx_21, 2*rank_2)

    # A22 B2
    u_idx_22 = [10, 11, 14, 15]
    v_idx_2 = [14,15,16,17,18,19,20, 21,22,23,24,25,26,27]
    w_idx_22 = []
    for k in range(7):
        w_idx_22.append(k*4 + 2)
        w_idx_22.append(k*4 + 3)
    add_block(u_idx_22, v_idx_2, w_idx_22, 3*rank_2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
