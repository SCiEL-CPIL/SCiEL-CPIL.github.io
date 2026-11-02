import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-50 decomposition for 2x4 x 4x7 matrix multiplication
    using block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build U, V, W matrices
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)  
    # W: (p*n, rank) = (14, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_factor(u_vec, v_vec, w_vec):
        """Add one rank-1 factor to the decomposition."""
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    def strassen_2x2x2(offset_u, offset_v, offset_w):
        """
        Add Strassen decomposition for a 2x2 * 2x2 block.
        offset_u: index in U for the 2x2 block of A (4 entries)
        offset_v: index in V for the 2x2 block of B (4 entries)
        offset_w: index in W for the 2x2 output block (4 entries)
        W convention: W[k*n + i, r] where n=2, so W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
        Actually W[k*n+i] where k is column index, i is row index.
        So W[0*2+0]=C[0,0], W[0*2+1]=C[1,0], W[1*2+0]=C[0,1], W[1*2+1]=C[1,1]
        """
        # Strassen formulas for 2x2 * 2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: A11, A12, A21, A22 -> offset_u + 0,1,2,3
        # B indices: B11, B12, B21, B22 -> offset_v + 0,1,2,3
        # C indices: C11, C12, C21, C22 -> offset_w + 0,1,2,3
        # But W convention is W[k*n+i] so C[i,j] -> W[j*n+i]
        # C11 -> W[0*2+0] = W[0]
        # C12 -> W[1*2+0] = W[2]
        # C21 -> W[0*2+1] = W[1]
        # C22 -> W[1*2+1] = W[3]
        
        # M1 = (A11 + A22) * (B11 + B22)
        u1 = [0]*8
        u1[offset_u + 0] = 1
        u1[offset_u + 3] = 1
        v1 = [0]*28
        v1[offset_v + 0] = 1
        v1[offset_v + 3] = 1
        w1 = [0]*14
        w1[offset_w + 0] = 1  # C11
        w1[offset_w + 3] = 1  # C22
        add_factor(u1, v1, w1)
        
        # M2 = (A21 + A22) * B11
        u2 = [0]*8
        u2[offset_u + 2] = 1
        u2[offset_u + 3] = 1
        v2 = [0]*28
        v2[offset_v + 0] = 1
        w2 = [0]*14
        w2[offset_w + 1] = 1  # C21
        w2[offset_w + 3] = -1  # C22
        add_factor(u2, v2, w2)
        
        # M3 = A11 * (B12 - B22)
        u3 = [0]*8
        u3[offset_u + 0] = 1
        v3 = [0]*28
        v3[offset_v + 1] = 1
        v3[offset_v + 3] = -1
        w3 = [0]*14
        w3[offset_w + 2] = 1  # C12
        w3[offset_w + 3] = 1  # C22
        add_factor(u3, v3, w3)
        
        # M4 = A22 * (B21 - B11)
        u4 = [0]*8
        u4[offset_u + 3] = 1
        v4 = [0]*28
        v4[offset_v + 2] = 1
        v4[offset_v + 0] = -1
        w4 = [0]*14
        w4[offset_w + 0] = 1  # C11
        w4[offset_w + 1] = 1  # C21
        add_factor(u4, v4, w4)
        
        # M5 = (A11 + A12) * B22
        u5 = [0]*8
        u5[offset_u + 0] = 1
        u5[offset_u + 1] = 1
        v5 = [0]*28
        v5[offset_v + 3] = 1
        w5 = [0]*14
        w5[offset_w + 0] = -1  # C11
        w5[offset_w + 2] = 1  # C12
        add_factor(u5, v5, w5)
        
        # M6 = (A21 - A11) * (B11 + B12)
        u6 = [0]*8
        u6[offset_u + 2] = 1
        u6[offset_u + 0] = -1
        v6 = [0]*28
        v6[offset_v + 0] = 1
        v6[offset_v + 1] = 1
        w6 = [0]*14
        w6[offset_w + 3] = 1  # C22
        add_factor(u6, v6, w6)
        
        # M7 = (A12 - A22) * (B21 + B22)
        u7 = [0]*8
        u7[offset_u + 1] = 1
        u7[offset_u + 3] = -1
        v7 = [0]*28
        v7[offset_v + 2] = 1
        v7[offset_v + 3] = 1
        w7 = [0]*14
        w7[offset_w + 0] = 1  # C11
        add_factor(u7, v7, w7)
    
    def schoolbook_2x2x1(offset_u, offset_v, offset_w):
        """
        Add schoolbook decomposition for a 2x2 * 2x1 block.
        C is 2x1, so C[0,0] and C[1,0].
        W convention: W[k*n+i] where n=2, k is column (only 0), i is row.
        W[0] = C[0,0], W[1] = C[1,0]
        """
        # C[0,0] = A11*B11 + A12*B21
        # C[1,0] = A21*B11 + A22*B21
        
        # Term 1: A11 * B11
        u1 = [0]*8
        u1[offset_u + 0] = 1
        v1 = [0]*28
        v1[offset_v + 0] = 1
        w1 = [0]*14
        w1[offset_w + 0] = 1  # C[0,0]
        add_factor(u1, v1, w1)
        
        # Term 2: A12 * B21
        u2 = [0]*8
        u2[offset_u + 1] = 1
        v2 = [0]*28
        v2[offset_v + 2] = 1
        w2 = [0]*14
        w2[offset_w + 0] = 1  # C[0,0]
        add_factor(u2, v2, w2)
        
        # Term 3: A21 * B11
        u3 = [0]*8
        u3[offset_u + 2] = 1
        v3 = [0]*28
        v3[offset_v + 0] = 1
        w3 = [0]*14
        w3[offset_w + 1] = 1  # C[1,0]
        add_factor(u3, v3, w3)
        
        # Term 4: A22 * B21
        u4 = [0]*8
        u4[offset_u + 3] = 1
        v4 = [0]*28
        v4[offset_v + 2] = 1
        w4 = [0]*14
        w4[offset_w + 1] = 1  # C[1,0]
        add_factor(u4, v4, w4)
    
    # Now build the full decomposition
    # A is 2x4: A = [A_left | A_right] where A_left, A_right are 2x2
    # B is 4x7: B = [B_top; B_bottom] where B_top, B_bottom are 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left: rows 0-1, cols 0-1 of A -> U indices 0,1,2,3
    # A_right: rows 0-1, cols 2-3 of A -> U indices 4,5,6,7
    # B_top: rows 0-1 of B, all 7 cols -> V indices 0-6
    # B_bottom: rows 2-3 of B, all 7 cols -> V indices 7-13
    
    # For A_left * B_top (2x2 * 2x7):
    # Split B_top into 2x2, 2x2, 2x2, 2x1
    # B_top cols 0-1: V indices 0,1,2,3 -> 2x2 block
    # B_top cols 2-3: V indices 4,5,6,7 -> wait, B_top is rows 0-1 of B
    # B is 4x7, indexed as B[j,k] where j=0..3, k=0..6
    # V index for B[j,k] is j*7 + k
    # B_top: j=0,1; B_bottom: j=2,3
    # B_top cols 0-1: V[0*7+0]=V[0], V[1*7+0]=V[7], V[0*7+1]=V[1], V[1*7+1]=V[8]
    # Hmm, this is getting complex. Let me re-index.
    
    # V index for B[j,k] = j*p + k = j*7 + k
    # B_top (j=0,1):
    #   col 0: V[0], V[7]
    #   col 1: V[1], V[8]
    #   col 2: V[2], V[9]
    #   col 3: V[3], V[10]
    #   col 4: V[4], V[11]
    #   col 5: V[5], V[12]
    #   col 6: V[6], V[13]
    # B_bottom (j=2,3):
    #   col 0: V[14], V[21]
    #   col 1: V[15], V[22]
    #   col 2: V[16], V[23]
    #   col 3: V[17], V[24]
    #   col 4: V[18], V[25]
    #   col 5: V[19], V[26]
    #   col 6: V[20], V[27]
    
    # For the 2x2 blocks of B_top:
    # Block 1 (cols 0-1): B[0,0]=V[0], B[1,0]=V[7], B[0,1]=V[1], B[1,1]=V[8]
    # In Strassen, we need B11, B12, B21, B22 for a 2x2 matrix
    # B11=B[0,0]=V[0], B12=B[0,1]=V[1], B21=B[1,0]=V[7], B22=B[1,1]=V[8]
    
    # This is getting messy. Let me use a helper to map Strassen offsets.
    
    # Actually, let me just directly construct with explicit indices.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_factor(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # Helper to create vectors
    def make_u(indices_vals):
        vec = [0]*8
        for idx, val in indices_vals:
            vec[idx] = val
        return vec
    
    def make_v(indices_vals):
        vec = [0]*28
        for idx, val in indices_vals:
            vec[idx] = val
        return vec
    
    def make_w(indices_vals):
        vec = [0]*14
        for idx, val in indices_vals:
            vec[idx] = val
        return vec
    
    # A_left * B_top: A_left is 2x2 (U indices 0-3), B_top is 2x7
    # B_top split into 3 2x2 blocks and 1 2x1 block
    # B_top block 1 (cols 0-1): B11=V[0], B12=V[1], B21=V[7], B22=V[8]
    # B_top block 2 (cols 2-3): B11=V[2], B12=V[3], B21=V[9], B22=V[10]
    # B_top block 3 (cols 4-5): B11=V[4], B12=V[5], B21=V[11], B22=V[12]
    # B_top block 4 (col 6): B11=V[6], B21=V[13]
    
    # W for C: C is 2x7, W[k*2+i] where k=col, i=row
    # C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3], ..., C[0,6]=W[12], C[1,6]=W[13]
    
    # Strassen for A_left * B_top_block1 (cols 0-1 of C)
    # Output: C[0,0], C[1,0], C[0,1], C[1,1] -> W[0], W[1], W[2], W[3]
    # M1 = (A11+A22)*(B11+B22) = (U0+U3)*(V0+V8)
    add_factor(make_u([(0,1),(3,1)]), make_v([(0,1),(8,1)]), make_w([(0,1),(3,1)]))
    # M2 = (A21+A22)*B11 = (U2+U3)*V0
    add_factor(make_u([(2,1),(3,1)]), make_v([(0,1)]), make_w([(1,1),(3,-1)]))
    # M3 = A11*(B12-B22) = U0*(V1-V8)
    add_factor(make_u([(0,1)]), make_v([(1,1),(8,-1)]), make_w([(2,1),(3,1)]))
    # M4 = A22*(B21-B11) = U3*(V7-V0)
    add_factor(make_u([(3,1)]), make_v([(7,1),(0,-1)]), make_w([(0,1),(1,1)]))
    # M5 = (A11+A12)*B22 = (U0+U1)*V8
    add_factor(make_u([(0,1),(1,1)]), make_v([(8,1)]), make_w([(0,-1),(2,1)]))
    # M6 = (A21-A11)*(B11+B12) = (U2-U0)*(V0+V1)
    add_factor(make_u([(2,1),(0,-1)]), make_v([(0,1),(1,1)]), make_w([(3,1)]))
    # M7 = (A12-A22)*(B21+B22) = (U1-U3)*(V7+V8)
    add_factor(make_u([(1,1),(3,-1)]), make_v([(7,1),(8,1)]), make_w([(0,1)]))
    
    # Strassen for A_left * B_top_block2 (cols 2-3 of C)
    # Output: C[0,2], C[1,2], C[0,3], C[1,3] -> W[4], W[5], W[6], W[7]
    # B11=V[2], B12=V[3], B21=V[9], B22=V[10]
    add_factor(make_u([(0,1),(3,1)]), make_v([(2,1),(10,1)]), make_w([(4,1),(7,1)]))
    add_factor(make_u([(2,1),(3,1)]), make_v([(2,1)]), make_w([(5,1),(7,-1)]))
    add_factor(make_u([(0,1)]), make_v([(3,1),(10,-1)]), make_w([(6,1),(7,1)]))
    add_factor(make_u([(3,1)]), make_v([(9,1),(2,-1)]), make_w([(4,1),(5,1)]))
    add_factor(make_u([(0,1),(1,1)]), make_v([(10,1)]), make_w([(4,-1),(6,1)]))
    add_factor(make_u([(2,1),(0,-1)]), make_v([(2,1),(3,1)]), make_w([(7,1)]))
    add_factor(make_u([(1,1),(3,-1)]), make_v([(9,1),(10,1)]), make_w([(4,1)]))
    
    # Strassen for A_left * B_top_block3 (cols 4-5 of C)
    # Output: C[0,4], C[1,4], C[0,5], C[1,5] -> W[8], W[9], W[10], W[11]
    # B11=V[4], B12=V[5], B21=V[11], B22=V[12]
    add_factor(make_u([(0,1),(3,1)]), make_v([(4,1),(12,1)]), make_w([(8,1),(11,1)]))
    add_factor(make_u([(2,1),(3,1)]), make_v([(4,1)]), make_w([(9,1),(11,-1)]))
    add_factor(make_u([(0,1)]), make_v([(5,1),(12,-1)]), make_w([(10,1),(11,1)]))
    add_factor(make_u([(3,1)]), make_v([(11,1),(4,-1)]), make_w([(8,1),(9,1)]))
    add_factor(make_u([(0,1),(1,1)]), make_v([(12,1)]), make_w([(8,-1),(10,1)]))
    add_factor(make_u([(2,1),(0,-1)]), make_v([(4,1),(5,1)]), make_w([(11,1)]))
    add_factor(make_u([(1,1),(3,-1)]), make_v([(11,1),(12,1)]), make_w([(8,1)]))
    
    # Schoolbook for A_left * B_top_block4 (col 6 of C)
    # Output: C[0,6], C[1,6] -> W[12], W[13]
    # B11=V[6], B21=V[13]
    # C[0,6] = A11*B11 + A12*B21 = U0*V6 + U1*V13
    # C[1,6] = A21*B11 + A22*B21 = U2*V6 + U3*V13
    add_factor(make_u([(0,1)]), make_v([(6,1)]), make_w([(12,1)]))
    add_factor(make_u([(1,1)]), make_v([(13,1)]), make_w([(12,1)]))
    add_factor(make_u([(2,1)]), make_v([(6,1)]), make_w([(13,1)]))
    add_factor(make_u([(3,1)]), make_v([(13,1)]), make_w([(13,1)]))
    
    # Now A_right * B_bottom: A_right is 2x2 (U indices 4-7), B_bottom is 2x7
    # B_bottom (j=2,3):
    # col 0: V[14], V[21]
    # col 1: V[15], V[22]
    # col 2: V[16], V[23]
    # col 3: V[17], V[24]
    # col 4: V[18], V[25]
    # col 5: V[19], V[26]
    # col 6: V[20], V[27]
    
    # Strassen for A_right * B_bottom_block1 (cols 0-1 of C)
    # B11=V[14], B12=V[15], B21=V[21], B22=V[22]
    add_factor(make_u([(4,1),(7,1)]), make_v([(14,1),(22,1)]), make_w([(0,1),(3,1)]))
    add_factor(make_u([(6,1),(7,1)]), make_v([(14,1)]), make_w([(1,1),(3,-1)]))
    add_factor(make_u([(4,1)]), make_v([(15,1),(22,-1)]), make_w([(2,1),(3,1)]))
    add_factor(make_u([(7,1)]), make_v([(21,1),(14,-1)]), make_w([(0,1),(1,1)]))
    add_factor(make_u([(4,1),(5,1)]), make_v([(22,1)]), make_w([(0,-1),(2,1)]))
    add_factor(make_u([(6,1),(4,-1)]), make_v([(14,1),(15,1)]), make_w([(3,1)]))
    add_factor(make_u([(5,1),(7,-1)]), make_v([(21,1),(22,1)]), make_w([(0,1)]))
    
    # Strassen for A_right * B_bottom_block2 (cols 2-3 of C)
    # B11=V[16], B12=V[17], B21=V[23], B22=V[24]
    add_factor(make_u([(4,1),(7,1)]), make_v([(16,1),(24,1)]), make_w([(4,1),(7,1)]))
    add_factor(make_u([(6,1),(7,1)]), make_v([(16,1)]), make_w([(5,1),(7,-1)]))
    add_factor(make_u([(4,1)]), make_v([(17,1),(24,-1)]), make_w([(6,1),(7,1)]))
    add_factor(make_u([(7,1)]), make_v([(23,1),(16,-1)]), make_w([(4,1),(5,1)]))
    add_factor(make_u([(4,1),(5,1)]), make_v([(24,1)]), make_w([(4,-1),(6,1)]))
    add_factor(make_u([(6,1),(4,-1)]), make_v([(16,1),(17,1)]), make_w([(7,1)]))
    add_factor(make_u([(5,1),(7,-1)]), make_v([(23,1),(24,1)]), make_w([(4,1)]))
    
    # Strassen for A_right * B_bottom_block3 (cols 4-5 of C)
    # B11=V[18], B12=V[19], B21=V[25], B22=V[26]
    add_factor(make_u([(4,1),(7,1)]), make_v([(18,1),(26,1)]), make_w([(8,1),(11,1)]))
    add_factor(make_u([(6,1),(7,1)]), make_v([(18,1)]), make_w([(9,1),(11,-1)]))
    add_factor(make_u([(4,1)]), make_v([(19,1),(26,-1)]), make_w([(10,1),(11,1)]))
    add_factor(make_u([(7,1)]), make_v([(25,1),(18,-1)]), make_w([(8,1),(9,1)]))
    add_factor(make_u([(4,1),(5,1)]), make_v([(26,1)]), make_w([(8,-1),(10,1)]))
    add_factor(make_u([(6,1),(4,-1)]), make_v([(18,1),(19,1)]), make_w([(11,1)]))
    add_factor(make_u([(5,1),(7,-1)]), make_v([(25,1),(26,1)]), make_w([(8,1)]))
    
    # Schoolbook for A_right * B_bottom_block4 (col 6 of C)
    # B11=V[20], B21=V[27]
    add_factor(make_u([(4,1)]), make_v([(20,1)]), make_w([(12,1)]))
    add_factor(make_u([(5,1)]), make_v([(27,1)]), make_w([(12,1)]))
    add_factor(make_u([(6,1)]), make_v([(20,1)]), make_w([(13,1)]))
    add_factor(make_u([(7,1)]), make_v([(27,1)]), make_w([(13,1)]))
    
    rank = len(U_list)
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
