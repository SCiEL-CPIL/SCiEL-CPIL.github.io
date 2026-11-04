import numpy as np

def construct():
    n, m, p = DIMENSIONS # 4, 4, 7
    
    # We will build the decomposition by combining smaller blocks.
    # Block 1: 4x4 * 4x4 (first 4 columns of B)
    # Block 2: 4x4 * 4x3 (last 3 columns of B)
    
    # Helper to get Strassen factors for 2x2x2
    def get_strassen_222():
        # U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices:
        # U: i*m+j -> 0..3
        # V: j*p+k -> 0..3
        # W: k*n+i -> 0..3
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # A11=0, A22=1 (in 2x2 block indices 0,1,2,3 -> A11=0, A12=1, A21=2, A22=3)
        # B11=0, B12=1, B21=2, B22=3
        # C11=0, C12=1, C21=2, C22=3
        
        # Col 0
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[2,0] = -1; W[3,0] = 1
        
        # M2 = (A21+A22)B11
        # Col 1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        
        # M3 = A11(B12-B22)
        # Col 2
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # M4 = A22(B21-B11)
        # Col 3
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # M5 = (A11+A12)B22
        # Col 4
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # Col 5
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # Col 6
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W

    # Helper to get schoolbook factors for 2x2x1
    def get_schoolbook_221():
        # U (4, 4), V (2, 4), W (1*2=2, 4) ? No, W is p*n. p=1, n=2. W is 2x4.
        # Wait, for 2x2x1:
        # n=2, m=2, p=1.
        # U: n*m = 4.
        # V: m*p = 2.
        # W: p*n = 2.
        # Rank 4.
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((2, 4), dtype=int)
        W = np.zeros((2, 4), dtype=int)
        
        for i in range(2):
            for j in range(2):
                col = i*2 + j
                U[i*2 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W

    # Combine 2x2x2 blocks to form 4x4x4
    # A is 4x4, B is 4x4.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need to map the local 2x2 indices to global 4x4 indices.
    # This is complex to do manually. 
    # Instead, let's just use the 2x2 Strassen for each of the 4 terms?
    # That would be 4 * 7 = 28.
    # But we know 4x4x4 can be done in 20.
    # The 20-rank scheme is a specific combination.
    
    # Given the complexity of hardcoding the 20-rank scheme, 
    # and the risk of error, let's stick to the block-wise 28+44=72 or similar.
    # Actually, the 4x4x4 rank 20 is standard.
    # Let's try to implement the 4x4x4 rank 20 scheme if possible.
    
    # Alternative: Use the 2x2 Strassen for each 2x2 block multiplication.
    # For 4x4x4:
    # C11 = A11 B11 + A12 B21.
    # We can use Strassen for A11 B11 (rank 7) and A12 B21 (rank 7). Total 14.
    # Similarly for other blocks.
    # Total 4 * 14 = 56?
    # Wait, A11 B11 is 2x2x2. Rank 7.
    # A12 B21 is 2x2x2. Rank 7.
    # So C11 takes 14.
    # C12 takes 14.
    # C21 takes 14.
    # C22 takes 14.
    # Total 56.
    # This is better than 112, but worse than 20.
    
    # For 4x4x3:
    # C11 (2x2x3) = A11 B11 + A12 B21.
    # A11 B11 is 2x2x2 (rank 7).
    # A12 B21 is 2x2x1? No, B11 is 2x2, B21 is 2x1?
    # If we split B (4x3) into rows 2+2.
    # B_top (2x3), B_bot (2x3).
    # C_top = A11 B_top + A12 B_bot.
    # A11 B_top is 2x2x3.
    # A12 B_bot is 2x2x3.
    # Rank of 2x2x3 is 11 (7 for 2x2x2 + 4 for 2x2x1).
    # So C_top takes 11 + 11 = 22?
    # No, we can share? No, they are separate multiplications.
    # So 4x4x3 takes 4 * 11 = 44.
    
    # Total rank: 56 (for 4x4x4) + 44 (for 4x4x3) = 100.
    # This is the same as my previous estimate.
    
    # Can we do better for 4x4x4?
    # Yes, 20 is known.
    # Can we do better for 4x4x3?
    # Maybe.
    
    # Let's implement the 56+44=100 scheme. It's safer than trying to hardcode 20.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(u, v, w, offset_u, offset_v, offset_w):
        # u, v, w are local matrices
        # offset_u, offset_v, offset_w are starting indices in global matrices
        # We need to append columns to global lists
        pass

    # This approach is getting messy with offsets.
    # Let's just construct the full U, V, W directly.
    
    # We will use the 2x2 Strassen (rank 7) for 2x2x2 parts.
    # And 2x2 Schoolbook (rank 4) for 2x2x1 parts.
    
    # 4x4x4 part (cols 0-3 of B)
    # B is 4x4. Split into 2x2 blocks.
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1).
    
    # 4x4x3 part (cols 4-6 of B)
    # B is 4x3. Split rows into 2+2.
    # B_top (rows 0-1, cols 0-2), B_bot (rows 2-3, cols 0-2).
    
    # This mixing of splits is hard to manage with a single U,V,W.
    
    # Let's stick to the simplest improvement:
    # Use Strassen for 2x2x2 (rank 7) everywhere possible.
    # For 4x4x7:
    # Split A into 2x2 blocks.
    # Split B into 4x2, 4x2, 4x2, 4x1.
    # For each 4x2 block of B:
    #   Split B_block into 2x2 top and 2x2 bot.
    #   A * B_block = [A11 B_top + A12 B_bot; A21 B_top + A22 B_bot]
    #   Each term is 2x2x2.
    #   Use Strassen for each. 4 terms * 7 rank = 28.
    # For 4x1 block of B:
    #   Split B_block into 2x1 top and 2x1 bot.
    #   A * B_block = [A11 B_top + A12 B_bot; A21 B_top + A22 B_bot]
    #   Each term is 2x2x1.
    #   Use Schoolbook for each. 4 terms * 4 rank = 16.
    # Total: 3 * 28 + 16 = 84 + 16 = 100.
    
    # Let's implement this.
    
    U = []
    V = []
    W = []
    
    # Pre-compute Strassen 2x2x2 factors
    U_s, V_s, W_s = get_strassen_222()
    # Pre-compute Schoolbook 2x2x1 factors
    U_b, V_b, W_b = get_schoolbook_221()
    
    # Global dimensions
    # U: 16 x Rank
    # V: 28 x Rank
    # W: 28 x Rank
    
    # We will build column by column.
    
    # For each of the 4 column groups in B:
    # Group 0, 1, 2: 2 columns each.
    # Group 3: 1 column.
    
    for group in range(4):
        if group < 3:
            # 2 columns
            # B block is 4x2.
            # Rows 0-1: B_top (2x2)
            # Rows 2-3: B_bot (2x2)
            
            # Term 1: A11 * B_top
            # A11 is rows 0-1, cols 0-1 of A.
            # B_top is rows 0-1, cols group*2 .. group*2+1 of B.
            
            # Map local 2x2x2 indices to global.
            # U local (4x7) -> Global U (16x7)
            # V local (4x7) -> Global V (28x7)
            # W local (4x7) -> Global W (28x7)
            
            for r in range(7):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                # A11 * B_top
                # A indices: 0,1,2,3 (local) -> Global 0,1,2,3
                # B indices: 0,1,2,3 (local) -> Global (group*2)*4 + 0..3?
                # Wait, B is 4x7.
                # B index j*p+k. p=7.
                # Row j (0-3), Col k (0-6).
                # B_top rows 0-1, cols group*2, group*2+1.
                # Local B index for B_top:
                # row 0, col 0 -> local 0
                # row 0, col 1 -> local 1
                # row 1, col 0 -> local 2
                # row 1, col 1 -> local 3
                
                # Global B index:
                # local 0 -> row 0, col group*2 -> 0*7 + group*2
                # local 1 -> row 0, col group*2+1 -> 0*7 + group*2+1
                # local 2 -> row 1, col group*2 -> 1*7 + group*2
                # local 3 -> row 1, col group*2+1 -> 1*7 + group*2+1
                
                # A11:
                # local 0 -> global 0
                # local 1 -> global 1
                # local 2 -> global 2
                # local 3 -> global 3
                
                # C output:
                # C is 4x7.
                # C_top (rows 0-1) comes from A11 B_top + A12 B_bot.
                # This term contributes to C rows 0-1.
                # Local C index:
                # local 0 -> row 0, col 0
                # local 1 -> row 0, col 1
                # local 2 -> row 1, col 0
                # local 3 -> row 1, col 1
                
                # Global C index:
                # local 0 -> row 0, col group*2 -> 0*4 + group*2
                # local 1 -> row 0, col group*2+1 -> 0*4 + group*2+1
                # local 2 -> row 1, col group*2 -> 1*4 + group*2
                # local 3 -> row 1, col group*2+1 -> 1*4 + group*2+1
                
                # Fill U
                for i in range(4):
                    if U_s[i, r] != 0:
                        u_col[i] += U_s[i, r] # A11 offset 0
                
                # Fill V
                v_map = [group*2, group*2+1, 7+group*2, 7+group*2+1]
                for j in range(4):
                    if V_s[j, r] != 0:
                        v_col[v_map[j]] += V_s[j, r]
                        
                # Fill W
                w_map = [group*2, group*2+1, 4+group*2, 4+group*2+1]
                for k in range(4):
                    if W_s[k, r] != 0:
                        w_col[w_map[k]] += W_s[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 2: A12 * B_bot
            # A12 is rows 0-1, cols 2-3 of A.
            # Global A indices: 4,5,6,7.
            # B_bot is rows 2-3, cols group*2..group*2+1.
            # Global B indices:
            # row 2, col g*2 -> 2*7 + g*2
            # row 2, col g*2+1 -> 2*7 + g*2+1
            # row 3, col g*2 -> 3*7 + g*2
            # row 3, col g*2+1 -> 3*7 + g*2+1
            
            for r in range(7):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [4,5,6,7]
                for i in range(4):
                    if U_s[i, r] != 0:
                        u_col[a_map[i]] += U_s[i, r]
                        
                v_map = [14+group*2, 14+group*2+1, 21+group*2, 21+group*2+1]
                for j in range(4):
                    if V_s[j, r] != 0:
                        v_col[v_map[j]] += V_s[j, r]
                        
                # W is same as before (C rows 0-1)
                w_map = [group*2, group*2+1, 4+group*2, 4+group*2+1]
                for k in range(4):
                    if W_s[k, r] != 0:
                        w_col[w_map[k]] += W_s[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 3: A21 * B_top
            # A21 is rows 2-3, cols 0-1.
            # Global A indices: 8,9,10,11.
            # B_top same as before.
            # C rows 2-3.
            
            for r in range(7):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [8,9,10,11]
                for i in range(4):
                    if U_s[i, r] != 0:
                        u_col[a_map[i]] += U_s[i, r]
                        
                v_map = [group*2, group*2+1, 7+group*2, 7+group*2+1]
                for j in range(4):
                    if V_s[j, r] != 0:
                        v_col[v_map[j]] += V_s[j, r]
                        
                w_map = [8+group*2, 8+group*2+1, 12+group*2, 12+group*2+1]
                for k in range(4):
                    if W_s[k, r] != 0:
                        w_col[w_map[k]] += W_s[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 4: A22 * B_bot
            # A22 is rows 2-3, cols 2-3.
            # Global A indices: 12,13,14,15.
            # B_bot same as before.
            # C rows 2-3.
            
            for r in range(7):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [12,13,14,15]
                for i in range(4):
                    if U_s[i, r] != 0:
                        u_col[a_map[i]] += U_s[i, r]
                        
                v_map = [14+group*2, 14+group*2+1, 21+group*2, 21+group*2+1]
                for j in range(4):
                    if V_s[j, r] != 0:
                        v_col[v_map[j]] += V_s[j, r]
                        
                w_map = [8+group*2, 8+group*2+1, 12+group*2, 12+group*2+1]
                for k in range(4):
                    if W_s[k, r] != 0:
                        w_col[w_map[k]] += W_s[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
        else:
            # Group 3: 1 column
            # B block is 4x1.
            # B_top (2x1), B_bot (2x1).
            
            # Term 1: A11 * B_top
            for r in range(4):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                # A11
                for i in range(4):
                    if U_b[i, r] != 0:
                        u_col[i] += U_b[i, r]
                        
                # B_top: row 0, col 6; row 1, col 6
                # Local V index 0 -> row 0, col 0 (local) -> Global 0*7+6 = 6
                # Local V index 1 -> row 1, col 0 (local) -> Global 1*7+6 = 13
                v_map = [6, 13]
                for j in range(2):
                    if V_b[j, r] != 0:
                        v_col[v_map[j]] += V_b[j, r]
                        
                # W: C rows 0-1, col 6
                # Local W index 0 -> row 0, col 0 -> Global 0*4+6 = 6
                # Local W index 1 -> row 1, col 0 -> Global 1*4+6 = 10
                w_map = [6, 10]
                for k in range(2):
                    if W_b[k, r] != 0:
                        w_col[w_map[k]] += W_b[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 2: A12 * B_bot
            for r in range(4):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [4,5,6,7]
                for i in range(4):
                    if U_b[i, r] != 0:
                        u_col[a_map[i]] += U_b[i, r]
                        
                # B_bot: row 2, col 6; row 3, col 6
                v_map = [2*7+6, 3*7+6]
                for j in range(2):
                    if V_b[j, r] != 0:
                        v_col[v_map[j]] += V_b[j, r]
                        
                w_map = [6, 10]
                for k in range(2):
                    if W_b[k, r] != 0:
                        w_col[w_map[k]] += W_b[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 3: A21 * B_top
            for r in range(4):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [8,9,10,11]
                for i in range(4):
                    if U_b[i, r] != 0:
                        u_col[a_map[i]] += U_b[i, r]
                        
                v_map = [6, 13]
                for j in range(2):
                    if V_b[j, r] != 0:
                        v_col[v_map[j]] += V_b[j, r]
                        
                # C rows 2-3, col 6
                w_map = [2*4+6, 3*4+6]
                for k in range(2):
                    if W_b[k, r] != 0:
                        w_col[w_map[k]] += W_b[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
            # Term 4: A22 * B_bot
            for r in range(4):
                u_col = np.zeros(16)
                v_col = np.zeros(28)
                w_col = np.zeros(28)
                
                a_map = [12,13,14,15]
                for i in range(4):
                    if U_b[i, r] != 0:
                        u_col[a_map[i]] += U_b[i, r]
                        
                v_map = [2*7+6, 3*7+6]
                for j in range(2):
                    if V_b[j, r] != 0:
                        v_col[v_map[j]] += V_b[j, r]
                        
                w_map = [2*4+6, 3*4+6]
                for k in range(2):
                    if W_b[k, r] != 0:
                        w_col[w_map[k]] += W_b[k, r]
                        
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)

    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
