import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We construct the decomposition for 4x4x4 and 4x4x2
    # and combine them.
    
    # Helper to get Strassen 2x2x2 tensors
    # Returns U, V, W as lists of vectors (length 4, 4, 4)
    def get_strassen_222():
        # Indices for 2x2: i in 0,1; j in 0,1; k in 0,1
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        U = []
        V = []
        W = []
        
        # P1 = (a+d)(e+h)
        # a:00, d:11 -> U: 1,0,0,1
        # e:00, h:11 -> V: 1,0,0,1
        # C11:00, C22:11 -> W: 1,0,0,1 (idx 0, 3)
        U.append([1,0,0,1])
        V.append([1,0,0,1])
        W.append([1,0,0,1])
        
        # P2 = (c+d)e
        # c:10, d:11 -> U: 0,0,1,1
        # e:00 -> V: 1,0,0,0
        # C21:10 -> W: 0,1,0,0 (idx 2)
        U.append([0,0,1,1])
        V.append([1,0,0,0])
        W.append([0,1,0,0])
        
        # P3 = a(f-h)
        # a:00 -> U: 1,0,0,0
        # f:01, h:11 -> V: 0,1,0,-1
        # C12:01 -> W: 0,0,1,0 (idx 1)
        U.append([1,0,0,0])
        V.append([0,1,0,-1])
        W.append([0,0,1,0])
        
        # P4 = d(g-e)
        # d:11 -> U: 0,0,0,1
        # g:10, e:00 -> V: 0,0,1,-1
        # C11:00, C21:10 -> W: 1,0,1,0 (idx 0, 2)
        U.append([0,0,0,1])
        V.append([0,0,1,-1])
        W.append([1,0,1,0])
        
        # P5 = (a+b)h
        # a:00, b:01 -> U: 1,1,0,0
        # h:11 -> V: 0,0,0,1
        # C11:00, C12:01 -> W: 1,1,0,0 (idx 0, 1) with -1 for C11?
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # So W5 has -1 at idx 0, +1 at idx 1.
        U.append([1,1,0,0])
        V.append([0,0,0,1])
        W.append([-1,1,0,0])
        
        # P6 = (c-a)(e+f)
        # c:10, a:00 -> U: 0,0,1,-1
        # e:00, f:01 -> V: 1,1,0,0
        # C22:11 -> W: 0,0,0,1 (idx 3)
        U.append([0,0,1,-1])
        V.append([1,1,0,0])
        W.append([0,0,0,1])
        
        # P7 = (b-d)(g+h)
        # b:01, d:11 -> U: 0,1,0,-1
        # g:10, h:11 -> V: 0,0,1,1
        # C11:00, C22:11 -> W: 1,0,0,1 (idx 0, 3)
        # C11 = ... + P7
        # C22 = ... + P6 (no P7 in C22? Wait)
        # C22 = P1 - P2 + P3 + P6
        # C11 = P1 + P4 - P5 + P7
        # So W7 has +1 at idx 0, 0 at idx 3?
        # Let's re-verify Strassen.
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        # Yes.
        U.append([0,1,0,-1])
        V.append([0,0,1,1])
        W.append([1,0,0,0])
        
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()
    
    # Build 4x4x4
    # A is 4x4. Blocks A11, A12, A21, A22.
    # B is 4x4. Blocks B11, B12, B21, B22.
    # C is 4x4. Blocks C11, C12, C21, C22.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A[i,j] where i,j in 0..3.
    # Block A11: i in 0,1; j in 0,1.
    # Block A12: i in 0,1; j in 2,3.
    # Block A21: i in 2,3; j in 0,1.
    # Block A22: i in 2,3; j in 2,3.
    
    # The 7 products are:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each product is a 2x2x2 multiplication.
    # We need to combine the U, V, W vectors.
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    def expand_U(vec_222, block_row, block_col):
        # vec_222 is length 4.
        # U index in 4x4 is i*4+j.
        # block_row, block_col in {0,1}.
        # i_local in 0,1 -> i_global = block_row*2 + i_local
        # j_local in 0,1 -> j_global = block_col*2 + j_local
        expanded = [0]*16
        for idx, val in enumerate(vec_222):
            i_l = idx // 2
            j_l = idx % 2
            i_g = block_row*2 + i_l
            j_g = block_col*2 + j_l
            expanded[i_g*4 + j_g] = val
        return expanded

    def expand_V(vec_222, block_row, block_col):
        # V index in 4x4 is j*4+k.
        # j_local in 0,1 -> j_global = block_row*2 + j_local
        # k_local in 0,1 -> k_global = block_col*2 + k_local
        expanded = [0]*16
        for idx, val in enumerate(vec_222):
            j_l = idx // 2
            k_l = idx % 2
            j_g = block_row*2 + j_l
            k_g = block_col*2 + k_l
            expanded[j_g*4 + k_g] = val
        return expanded

    def expand_W(vec_222, block_row, block_col):
        # W index in 4x4 is k*4+i.
        # k_local in 0,1 -> k_global = block_row*2 + k_local
        # i_local in 0,1 -> i_global = block_col*2 + i_local
        expanded = [0]*16
        for idx, val in enumerate(vec_222):
            k_l = idx // 2
            i_l = idx % 2
            k_g = block_row*2 + k_l
            i_g = block_col*2 + i_l
            expanded[k_g*4 + i_g] = val
        return expanded

    # Define the 7 block products
    # (U_blocks, V_blocks, W_blocks)
    # U_blocks: list of (block_row, block_col) for A
    # V_blocks: list of (block_row, block_col) for B
    # W_blocks: list of (block_row, block_col) for C
    
    # P1 = (A11+A22)(B11+B22)
    # U: A11, A22. V: B11, B22. W: C11, C22.
    # We need to sum the contributions.
    # For each of the 7 rank-1 tensors in 222, we add contributions from the block sums.
    
    # This is easier: iterate over the 7 Strassen tensors.
    # For each Strassen tensor (u,v,w), we add 4 terms for P1?
    # No, P1 is one product. The Strassen tensor decomposes P1.
    # P1 = (A11+A22)(B11+B22).
    # This is not a single 2x2x2 multiplication.
    # It is the product of sum of matrices.
    # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
    # So P1 is a 2x2x2 multiplication of the summed matrices.
    # The Strassen decomposition applies to the 2x2x2 multiplication.
    # So for each of the 7 rank-1 tensors of the 2x2x2 Strassen algorithm,
    # we need to map the "input" A to A11+A22, etc.
    
    # How to map "input A" to A11+A22?
    # The U vector for the 2x2x2 multiplication has entries for A.
    # If the entry is 1 for A[0,0], it contributes to A11[0,0] and A22[0,0].
    # So we expand the U vector to 4x4 by placing the value at the corresponding positions in A11 and A22.
    
    def expand_sum_U(vec_222, blocks):
        # blocks: list of (br, bc)
        expanded = [0]*16
        for br, bc in blocks:
            e = expand_U(vec_222, br, bc)
            for i in range(16):
                expanded[i] += e[i]
        return expanded
        
    def expand_sum_V(vec_222, blocks):
        expanded = [0]*16
        for br, bc in blocks:
            e = expand_V(vec_222, br, bc)
            for i in range(16):
                expanded[i] += e[i]
        return expanded
        
    def expand_sum_W(vec_222, blocks):
        expanded = [0]*16
        for br, bc in blocks:
            e = expand_W(vec_222, br, bc)
            for i in range(16):
                expanded[i] += e[i]
        return expanded

    # P1: A_blocks=[(0,0),(1,1)], B_blocks=[(0,0),(1,1)], C_blocks=[(0,0),(1,1)]
    # P2: A_blocks=[(1,0),(1,1)], B_blocks=[(0,0)], C_blocks=[(1,0)]
    # P3: A_blocks=[(0,0)], B_blocks=[(0,1),(1,1)], C_blocks=[(0,1)]
    # P4: A_blocks=[(1,1)], B_blocks=[(1,0),(0,0)], C_blocks=[(0,0),(1,0)]
    # P5: A_blocks=[(0,0),(0,1)], B_blocks=[(1,1)], C_blocks=[(0,0),(0,1)]
    # P6: A_blocks=[(1,0),(0,0)], B_blocks=[(0,0),(0,1)], C_blocks=[(1,1)]
    # P7: A_blocks=[(0,1),(1,1)], B_blocks=[(1,0),(1,1)], C_blocks=[(0,0)]
    
    # Note: Signs in Strassen.
    # P3 = A11(B12-B22). B blocks are (0,1) and (1,1) with sign -1 for (1,1).
    # So expand_sum_V needs to handle signs.
    
    def expand_sum_signed(vec_222, blocks):
        # blocks: list of (br, bc, sign)
        expanded = [0]*16
        for br, bc, sign in blocks:
            e = expand_V(vec_222, br, bc) # Assuming V for now, need generic
            for i in range(16):
                expanded[i] += sign * e[i]
        return expanded
        
    # Let's make generic expand functions that take a type 'U', 'V', 'W'
    def expand_generic(vec_222, blocks, type):
        expanded = [0]*16
        for br, bc, sign in blocks:
            if type == 'U':
                e = expand_U(vec_222, br, bc)
            elif type == 'V':
                e = expand_V(vec_222, br, bc)
            else:
                e = expand_W(vec_222, br, bc)
            for i in range(16):
                expanded[i] += sign * e[i]
        return expanded

    # P1
    A_bl = [(0,0,1), (1,1,1)]
    B_bl = [(0,0,1), (1,1,1)]
    C_bl = [(0,0,1), (1,1,1)]
    # P2
    A_bl = [(1,0,1), (1,1,1)]
    B_bl = [(0,0,1)]
    C_bl = [(1,0,1)]
    # P3
    A_bl = [(0,0,1)]
    B_bl = [(0,1,1), (1,1,-1)]
    C_bl = [(0,1,1)]
    # P4
    A_bl = [(1,1,1)]
    B_bl = [(1,0,1), (0,0,-1)]
    C_bl = [(0,0,1), (1,0,1)]
    # P5
    A_bl = [(0,0,1), (0,1,1)]
    B_bl = [(1,1,1)]
    C_bl = [(0,0,-1), (0,1,1)]
    # P6
    A_bl = [(1,0,1), (0,0,-1)]
    B_bl = [(0,0,1), (0,1,1)]
    C_bl = [(1,1,1)]
    # P7
    A_bl = [(0,1,1), (1,1,-1)]
    B_bl = [(1,0,1), (1,1,1)]
    C_bl = [(0,0,1)]
    
    # Wait, P7 C_block is C11?
    # C11 = P1 + P4 - P5 + P7. Yes.
    
    products = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for u2, v2, w2 in zip(U_222, V_222, W_222):
        for a_bl, b_bl, c_bl in products:
            U_444.append(expand_generic(u2, a_bl, 'U'))
            V_444.append(expand_generic(v2, b_bl, 'V'))
            W_444.append(expand_generic(w2, c_bl, 'W'))
            
    # Now 4x4x2
    # A is 4x4. Split into A1 (2x4), A2 (2x4).
    # B is 4x2. Split into B1 (2x2), B2 (2x2).
    # C is 4x2. Split into C1 (2x2), C2 (2x2).
    # C1 = A1 B1 + A1 B2? No.
    # A1 is 2x4. B is 4x2.
    # A1 = [A1L, A1R] (2x2 each).
    # B = [B1T; B1B] (2x2 each).
    # C1 = A1L B1T + A1R B1B.
    # This is sum of two 2x2x2 multiplications.
    
    # We need to map 2x2x2 tensors to 4x4x2 tensors.
    # U index in 4x4: i*4+j.
    # V index in 4x2: j*2+k.
    # W index in 2x4: k*2+i.
    
    # A1L: i in 0,1; j in 0,1.
    # A1R: i in 0,1; j in 2,3.
    # B1T: j in 0,1; k in 0,1.
    # B1B: j in 2,3; k in 0,1.
    # C1: i in 0,1; k in 0,1.
    
    # For the first term (A1L B1T):
    # U: A1L. V: B1T. W: C1.
    # For the second term (A1R B1B):
    # U: A1R. V: B1B. W: C1.
    
    # We add these to the existing U, V, W.
    
    def expand_442_U(vec_222, block_col_A):
        # block_col_A: 0 for L, 1 for R
        expanded = [0]*16
        for idx, val in enumerate(vec_222):
            i_l = idx // 2
            j_l = idx % 2
            i_g = i_l # A1 is top half
            j_g = block_col_A*2 + j_l
            expanded[i_g*4 + j_g] = val
        return expanded
        
    def expand_442_V(vec_222, block_row_B):
        # block_row_B: 0 for T, 1 for B
        expanded = [0]*8 # 4x2 = 8
        for idx, val in enumerate(vec_222):
            j_l = idx // 2
            k_l = idx % 2
            j_g = block_row_B*2 + j_l
            k_g = k_l
            expanded[j_g*2 + k_g] = val
        return expanded
        
    def expand_442_W(vec_222):
        expanded = [0]*8 # 2x4 = 8
        for idx, val in enumerate(vec_222):
            k_l = idx // 2
            i_l = idx % 2
            k_g = k_l
            i_g = i_l
            expanded[k_g*2 + i_g] = val
        return expanded

    # C1 = A1L B1T + A1R B1B
    # C2 = A2L B2T + A2R B2B
    
    # A2L: i in 2,3; j in 0,1.
    # A2R: i in 2,3; j in 2,3.
    # B2T: j in 0,1; k in 0,1. (Same as B1T? No, B is 4x2. B1 is top 2 rows, B2 is bottom 2 rows? No, B is split into B1, B2 which are 2x2. So B1 is rows 0,1? No, B is 4x2. B1 is rows 0,1? No, B1 is cols 0,1? No, B is 4x2.
    # Wait, B is 4x2. Split into B1 (2x2) and B2 (2x2).
    # B1 is rows 0,1? Or cols?
    # A1 (2x4) * B (4x2).
    # A1 = [A1L, A1R]. A1L is cols 0,1. A1R is cols 2,3.
    # B = [B1T; B1B]. B1T is rows 0,1. B1B is rows 2,3.
    # So B1T is 2x2, B1B is 2x2.
    # Yes.
    
    # So for C1:
    # Term 1: A1L (cols 0,1) * B1T (rows 0,1).
    # Term 2: A1R (cols 2,3) * B1B (rows 2,3).
    
    # For C2:
    # Term 1: A2L (cols 0,1) * B1T (rows 0,1).
    # Term 2: A2R (cols 2,3) * B1B (rows 2,3).
    
    # Wait, B is the same for C1 and C2.
    # C1 = A1 B. C2 = A2 B.
    # So we use the same B1T, B1B for both.
    
    # We add 4 sets of Strassen tensors for 442.
    # 1. A1L B1T -> C1
    # 2. A1R B1B -> C1
    # 3. A2L B1T -> C2
    # 4. A2R B1B -> C2
    
    # We need to expand U to 16, V to 8, W to 8.
    # But our U_444 is already 16.
    # V_444 is 16 (for 4x4). We need V for 4x2.
    # So we create separate lists for the 442 part, then combine.
    
    U_442 = []
    V_442 = []
    W_442 = []
    
    for u2, v2, w2 in zip(U_222, V_222, W_222):
        # 1. A1L B1T -> C1
        U_442.append(expand_442_U(u2, 0))
        V_442.append(expand_442_V(v2, 0))
        W_442.append(expand_442_W(w2))
        
        # 2. A1R B1B -> C1
        U_442.append(expand_442_U(u2, 1))
        V_442.append(expand_442_V(v2, 1))
        W_442.append(expand_442_W(w2))
        
        # 3. A2L B1T -> C2
        # A2L: i in 2,3; j in 0,1.
        # expand_442_U needs to handle row offset.
        def expand_442_U_row(vec_222, block_col_A, row_offset):
            expanded = [0]*16
            for idx, val in enumerate(vec_222):
                i_l = idx // 2
                j_l = idx % 2
                i_g = row_offset + i_l
                j_g = block_col_A*2 + j_l
                expanded[i_g*4 + j_g] = val
            return expanded
            
        # Re-do 1 and 2 with row_offset 0
        # Remove last 2 added? No, just add 3 and 4.
        # But 1 and 2 were added with row_offset 0 implicitly?
        # My expand_442_U used i_g = i_l. So row_offset 0. Correct.
        
        U_442.append(expand_442_U_row(u2, 0, 2))
        V_442.append(expand_442_V(v2, 0))
        # W for C2: i in 2,3.
        def expand_442_W_row(vec_222, row_offset):
            expanded = [0]*8
            for idx, val in enumerate(vec_222):
                k_l = idx // 2
                i_l = idx % 2
                k_g = k_l
                i_g = row_offset + i_l
                expanded[k_g*2 + i_g] = val
            return expanded
        W_442.append(expand_442_W_row(w2, 2))
        
        # 4. A2R B1B -> C2
        U_442.append(expand_442_U_row(u2, 1, 2))
        V_442.append(expand_442_V(v2, 1))
        W_442.append(expand_442_W_row(w2, 2))
        
    # Combine 444 and 442
    # U_444 is 16x49. U_442 is 16x28.
    # V_444 is 16x49. V_442 is 8x28.
    # W_444 is 16x49. W_442 is 8x28.
    
    # Final V is 24x77.
    # V_444 corresponds to B (4x4). V_442 corresponds to B (4x2).
    # B is 4x6.
    # V index j*6+k.
    # For 444: j in 0..3, k in 0..3. Index j*6+k.
    # For 442: j in 0..3, k in 4..5. Index j*6+k.
    
    # We need to map V_444 (16) to 24.
    # V_444 index j*4+k.
    # New index j*6+k.
    
    def map_V_444_to_24(vec_16):
        expanded = [0]*24
        for idx, val in enumerate(vec_16):
            j = idx // 4
            k = idx % 4
            expanded[j*6 + k] = val
        return expanded
        
    def map_V_442_to_24(vec_8):
        expanded = [0]*24
        for idx, val in enumerate(vec_8):
            j = idx // 2
            k = idx % 2
            expanded[j*6 + (4+k)] = val
        return expanded
        
    def map_W_444_to_24(vec_16):
        # W index k*4+i.
        # New index k*4+i. (p=6, n=4).
        # W shape 24.
        # k in 0..5. i in 0..3.
        # For 444: k in 0..3.
        expanded = [0]*24
        for idx, val in enumerate(vec_16):
            k = idx // 4
            i = idx % 4
            expanded[k*4 + i] = val
        return expanded
        
    def map_W_442_to_24(vec_8):
        # W index k*2+i.
        # For 442: k in 0..1.
        # New index k*4+i? No.
        # W index is k*n+i. n=4.
        # For 442, k is 0..1 (from 4x2 B? No, B is 4x2. k is col of B. 0,1).
        # So k in 0..1.
        # New index k*4+i.
        expanded = [0]*24
        for idx, val in enumerate(vec_8):
            k = idx // 2
            i = idx % 2
            expanded[k*4 + i] = val
        return expanded
        
    U_final = U_444 + U_442
    V_final = [map_V_444_to_24(v) for v in V_444] + [map_V_442_to_24(v) for v in V_442]
    W_final = [map_W_444_to_24(w) for w in W_444] + [map_W_442_to_24(w) for w in W_442]
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    res = construct()
    print(len(res["U"][0]))
