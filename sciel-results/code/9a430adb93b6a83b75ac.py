import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # We will build the decomposition for 2x2x2 and 2x2x1, then combine.
    
    # Strassen 2x2x2 decomposition
    # U, V, W are (4, 7), (4, 7), (4, 7)
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # Indices for A: 0,1,2,3 (row-major 2x2)
    # Indices for B: 0,1,2,3 (row-major 2x2)
    # Indices for C: 0,1,2,3 (row-major 2x2)
    
    # M1 = (a00+a11)(b00+b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # Z11, Z22
    # Wait, W index is k*n+i.
    # C = [[Z11, Z12], [Z21, Z22]]
    # W[k*n+i, c]
    # Z11 is i=0, k=0 -> index 0
    # Z12 is i=0, k=1 -> index 1
    # Z21 is i=1, k=0 -> index 2
    # Z22 is i=1, k=1 -> index 3
    
    # Let's re-map W indices properly.
    # W[k*n + i, c]
    # Z11: k=0, i=0 -> 0
    # Z12: k=1, i=0 -> 1
    # Z21: k=0, i=1 -> 2
    # Z22: k=1, i=1 -> 3
    
    # M1 contributes to Z11 (+1) and Z22 (+1)
    W_s[0, 0] = 1
    W_s[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1 # Z21
    W_s[3, 1] = -1 # Z22
    
    # M3 = a00(b01-b11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1 # Z12
    W_s[3, 2] = 1 # Z22
    
    # M4 = a11(b10-b00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1 # Z11
    W_s[2, 3] = 1 # Z21
    
    # M5 = (a00+a01)b11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1 # Z11
    W_s[1, 4] = 1 # Z12
    
    # M6 = (a10-a00)(b00+b01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1 # Z22
    
    # M7 = (a01-a11)(b10+b11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # Z11
    W_s[2, 6] = 1 # Z21

    # Schoolbook 2x2x1 decomposition
    # A (2x2) * B (2x1) -> C (2x1)
    # C0 = a00*b00 + a01*b10
    # C1 = a10*b00 + a11*b10
    # W indices: C0 is i=0, k=0 -> 0. C1 is i=1, k=0 -> 1.
    U_b = np.zeros((4, 4), dtype=int)
    V_b = np.zeros((2, 4), dtype=int) # B is 2x1, so 2 elements
    W_b = np.zeros((2, 4), dtype=int) # C is 2x1, so 2 elements
    
    # M1 = a00*b00
    U_b[0, 0] = 1
    V_b[0, 0] = 1
    W_b[0, 0] = 1
    
    # M2 = a01*b10
    U_b[1, 1] = 1
    V_b[1, 1] = 1
    W_b[0, 1] = 1
    
    # M3 = a10*b00
    U_b[2, 2] = 1
    V_b[0, 2] = 1
    W_b[1, 2] = 1
    
    # M4 = a11*b10
    U_b[3, 3] = 1
    V_b[1, 3] = 1
    W_b[1, 3] = 1

    # Now combine for 2x2x7
    # B is 2x7. Split into B1 (2x4) and B2 (2x3).
    # B1 split into B1a (2x2) and B1b (2x2).
    # B2 split into B2a (2x2) and B2b (2x1).
    
    # Total rank for 2x2x7: 7 (B1a) + 7 (B1b) + 7 (B2a) + 4 (B2b) = 25.
    
    U_227 = np.zeros((4, 25), dtype=int)
    V_227 = np.zeros((2, 25), dtype=int) # B is 2 rows
    W_227 = np.zeros((2, 25), dtype=int) # C is 2 rows? No, C is 2x7.
    # W shape for 2x2x7: p*n = 7*2 = 14.
    # W[k*n+i, c].
    
    # We need to map the local W indices to global W indices.
    # For B1a (cols 0,1): C cols 0,1.
    # For B1b (cols 2,3): C cols 2,3.
    # For B2a (cols 4,5): C cols 4,5.
    # For B2b (col 6): C col 6.
    
    def add_block(U_block, V_block, W_block, col_offset, row_offset_C, row_offset_B):
        # U_block: (4, r)
        # V_block: (2, r)
        # W_block: (r, 4) or (4, r)? 
        # In my definition above, W_s was (4, 7) i.e. (C_elements, rank).
        # But interface requires W[p*n, rank].
        # p=2, n=2. p*n=4.
        # So W_block is (4, r).
        # W_block[c_idx, r_idx].
        # c_idx corresponds to k*n+i.
        
        # U is shared, so we just copy.
        U_227[:, col_offset:col_offset+U_block.shape[1]] = U_block
        
        # V needs row offset? No, V is (m*p, rank) for the whole problem?
        # For 2x2x7, m=2, p=7. V is (14, rank).
        # Here we are building 2x2x7.
        # V is (2*7, 25) = (14, 25).
        # B is 2x7.
        # V[j*p+k, c].
        # j is row of B (0,1). k is col of B (0..6).
        
        # V_block is (2, r).
        # We need to place it in V_227.
        # For B1a (cols 0,1):
        # V_block[0, r] -> V_227[0*7+0, col_offset+r] and V_227[0*7+1, col_offset+r]
        # V_block[1, r] -> V_227[1*7+0, col_offset+r] and V_227[1*7+1, col_offset+r]
        
        for r in range(U_block.shape[1]):
            # V_block[0, r] is B[0, col]
            # V_block[1, r] is B[1, col]
            # We need to map to V_227[j*7+k]
            
            # This is getting complicated to hardcode offsets.
            # Let's just build the full 2x4x7 directly.
            pass

    # Let's restart the construction for 2x4x7 directly.
    # Rank 50.
    # U: (8, 50)
    # V: (28, 50)
    # W: (14, 50)
    
    U = np.zeros((8, 50), dtype=int)
    V = np.zeros((28, 50), dtype=int)
    W = np.zeros((14, 50), dtype=int)
    
    # We have 4 blocks of B:
    # B1 (cols 0,1) -> 2x2x2 (Rank 7)
    # B2 (cols 2,3) -> 2x2x2 (Rank 7)
    # B3 (cols 4,5) -> 2x2x2 (Rank 7)
    # B4 (col 6) -> 2x2x1 (Rank 4)
    # Wait, B is 4x7.
    # A is 2x4.
    # A = [A1, A2] (2x2 each).
    # B = [B1, B2] (4x2, 4x2) and B3 (4x3)?
    # No, the split was A1*B1 + A2*B2.
    # B1 is 2x7. B2 is 2x7.
    # B1 split into 2x2, 2x2, 2x2, 2x1.
    # B2 split into 2x2, 2x2, 2x2, 2x1.
    
    # So we have 8 blocks of 2x2x2 and 2 blocks of 2x2x1.
    # Total Rank: 8*7 + 2*4 = 56 + 8 = 64?
    # No, the hint said 50.
    # Where is the saving?
    # Maybe B1 and B2 can be handled more efficiently?
    # Or maybe my split of B1 into 2x2,2x2,2x2,2x1 is suboptimal.
    # R(2,2,7) = 25.
    # 25 = 7+7+7+4? 25.
    # So 2 * 25 = 50.
    # Yes.
    
    # So we need to add 8 Strassen blocks and 2 Schoolbook blocks.
    
    def add_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        # a_offset: row offset in U (0 or 4)
        # b_offset: row offset in V (0 or 14)
        # c_offset: row offset in W (0 or 7)
        # col_start: column start in B (0, 2, 4, 6)
        
        for r in range(7):
            # U
            if U_s[0, r] != 0: U[a_offset, col_start+r] = U_s[0, r]
            if U_s[1, r] != 0: U[a_offset+1, col_start+r] = U_s[1, r]
            if U_s[2, r] != 0: U[a_offset+2, col_start+r] = U_s[2, r]
            if U_s[3, r] != 0: U[a_offset+3, col_start+r] = U_s[3, r]
            
            # V
            # V is (28, 50).
            # B is 4x7.
            # For A1 (rows 0,1 of A), we use rows 0,1 of B.
            # For A2 (rows 2,3 of A), we use rows 2,3 of B.
            # So b_offset is 0 for A1, 14 for A2?
            # V index: j*7+k.
            # j=0,1 for top half. j=2,3 for bottom half.
            # So offset 0 and 14.
            
            if V_s[0, r] != 0: V[b_offset + 0*7 + col_start, col_start+r] = V_s[0, r] # Wait, col index in V is k.
            # V_s is (4, 7). V_s[j, r] where j is 0..3 (2x2).
            # For top half (j=0,1): V[0*7+k] and V[1*7+k].
            # V_s[0] -> B[0,0], V_s[1] -> B[0,1], V_s[2] -> B[1,0], V_s[3] -> B[1,1].
            # So V_s[0] maps to V[b_offset + 0*7 + col_start]
            # V_s[1] maps to V[b_offset + 0*7 + col_start+1]
            # V_s[2] maps to V[b_offset + 1*7 + col_start]
            # V_s[3] maps to V[b_offset + 1*7 + col_start+1]
            
            if V_s[0, r] != 0: V[b_offset + 0*7 + col_start, col_start+r] = V_s[0, r]
            if V_s[1, r] != 0: V[b_offset + 0*7 + col_start+1, col_start+r] = V_s[1, r]
            if V_s[2, r] != 0: V[b_offset + 1*7 + col_start, col_start+r] = V_s[2, r]
            if V_s[3, r] != 0: V[b_offset + 1*7 + col_start+1, col_start+r] = V_s[3, r]
            
            # W
            # W is (14, 50).
            # C is 2x7.
            # W[k*n+i, c].
            # For A1, C rows 0,1.
            # For A2, C rows 0,1 (same C!).
            # So c_offset is 0 for both?
            # No, C is the same for both parts.
            # So we just add to the same W columns.
            
            # W_s is (4, 7).
            # W_s[0] -> C[0,0] (i=0, k=0) -> index 0
            # W_s[1] -> C[0,1] (i=0, k=1) -> index 1
            # W_s[2] -> C[1,0] (i=1, k=0) -> index 2
            # W_s[3] -> C[1,1] (i=1, k=1) -> index 3
            
            # Global C col is col_start.
            # So C[0, col_start] -> index 0*7 + col_start? No.
            # W index is k*n+i.
            # k is col of C (0..6). i is row of C (0..1).
            # So index is k*2+i.
            
            # W_s[0] (Z11) -> i=0, k=0 -> 0*2+0 = 0. Global: col_start*2 + 0.
            # W_s[1] (Z12) -> i=0, k=1 -> 1*2+0 = 2. Global: (col_start+1)*2 + 0.
            # W_s[2] (Z21) -> i=1, k=0 -> 0*2+1 = 1. Global: col_start*2 + 1.
            # W_s[3] (Z22) -> i=1, k=1 -> 1*2+1 = 3. Global: (col_start+1)*2 + 1.
            
            if W_s[0, r] != 0: W[col_start*2 + 0, col_start+r] += W_s[0, r]
            if W_s[1, r] != 0: W[(col_start+1)*2 + 0, col_start+r] += W_s[1, r]
            if W_s[2, r] != 0: W[col_start*2 + 1, col_start+r] += W_s[2, r]
            if W_s[3, r] != 0: W[(col_start+1)*2 + 1, col_start+r] += W_s[3, r]

    def add_schoolbook(U, V, W, a_offset, b_offset, c_offset, col_start):
        for r in range(4):
            if U_b[0, r] != 0: U[a_offset, col_start+r] = U_b[0, r]
            if U_b[1, r] != 0: U[a_offset+1, col_start+r] = U_b[1, r]
            if U_b[2, r] != 0: U[a_offset+2, col_start+r] = U_b[2, r]
            if U_b[3, r] != 0: U[a_offset+3, col_start+r] = U_b[3, r]
            
            if V_b[0, r] != 0: V[b_offset + 0*7 + col_start, col_start+r] = V_b[0, r]
            if V_b[1, r] != 0: V[b_offset + 1*7 + col_start, col_start+r] = V_b[1, r]
            
            # W
            # C is 2x1.
            # W_b[0] -> C[0,0] -> i=0, k=0 -> 0. Global: col_start*2 + 0.
            # W_b[1] -> C[1,0] -> i=1, k=0 -> 1. Global: col_start*2 + 1.
            
            if W_b[0, r] != 0: W[col_start*2 + 0, col_start+r] += W_b[0, r]
            if W_b[1, r] != 0: W[col_start*2 + 1, col_start+r] += W_b[1, r]

    # A1 (offset 0), B1 (offset 0)
    add_strassen(U, V, W, 0, 0, 0, 0)
    add_strassen(U, V, W, 0, 0, 0, 2)
    add_strassen(U, V, W, 0, 0, 0, 4)
    add_schoolbook(U, V, W, 0, 0, 0, 6)
    
    # A2 (offset 4), B2 (offset 14)
    add_strassen(U, V, W, 4, 14, 0, 0)
    add_strassen(U, V, W, 4, 14, 0, 2)
    add_strassen(U, V, W, 4, 14, 0, 4)
    add_schoolbook(U, V, W, 4, 14, 0, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

