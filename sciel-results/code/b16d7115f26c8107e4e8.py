"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Structure:
    - Two 4x4x4 blocks (columns 0-3 and 4-7 of B) using recursive Strassen (rank 49 each)
    - One 4x4x1 block (column 8 of B) using schoolbook (rank 16)
    Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen 2x2: 7 rank-1 terms
    # Indices: x00=0, x01=1, x10=2, x11=3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # M5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # For 4x4, view as 2x2 blocks of 2x2 matrices:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    #
    # Strassen at block level:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Each Pi is a 2x2 * 2x2 product, done via Strassen (7 terms each)
    # Total: 7 * 7 = 49 terms
    
    # Map 4x4 indices to block structure:
    # A indices: i in 0..3, j in 0..3
    #   A00: i in {0,1}, j in {0,1} -> local (i%2, j%2)
    #   A01: i in {0,1}, j in {2,3} -> local (i%2, (j-2)%2)
    #   A10: i in {2,3}, j in {0,1} -> local ((i-2)%2, j%2)
    #   A11: i in {2,3}, j in {2,3} -> local ((i-2)%2, (j-2)%2)
    #
    # For U: index is i*m + j = i*4 + j
    # For V: index is j*p + k, but within a 4x4 block, k is local 0..3
    # For W: index is k*n + i = k*4 + i
    
    def get_4x4_strassen_terms():
        """
        Generate the 49 rank-1 terms for 4x4 matrix multiplication.
        Each term is (u_vec, v_vec, w_vec) where each is a dict mapping index->value.
        """
        terms = []
        
        # Define the 7 Strassen block products with their sign patterns
        # For each block product, we need:
        # - Which A-blocks contribute and with what signs
        # - Which B-blocks contribute and with what signs
        # - Which C-blocks the result contributes to and with what signs
        
        # Block product definitions:
        # (a_blocks: list of (block_id, sign)), (b_blocks: list of (block_id, sign)), (c_blocks: list of (block_id, sign))
        # Block IDs: 0=A00, 1=A01, 2=A10, 3=A11 for A
        #             0=B00, 1=B01, 2=B10, 3=B11 for B
        #             0=C00, 1=C01, 2=C10, 3=C11 for C
        
        strassen_blocks = [
            # P1 = (A00+A11)*(B00+B11)
            {
                'a': [(0, 1), (3, 1)],
                'b': [(0, 1), (3, 1)],
                'c': [(0, 1), (3, 1)],  # contributes to C00 and C11
            },
            # P2 = (A10+A11)*B00
            {
                'a': [(2, 1), (3, 1)],
                'b': [(0, 1)],
                'c': [(2, 1), (3, -1)],  # C10 += P2, C11 -= P2
            },
            # P3 = A00*(B01-B11)
            {
                'a': [(0, 1)],
                'b': [(1, 1), (3, -1)],
                'c': [(1, 1), (3, 1)],  # C01 += P3, C11 += P3
            },
            # P4 = A11*(B10-B00)
            {
                'a': [(3, 1)],
                'b': [(2, 1), (0, -1)],
                'c': [(0, 1), (2, 1)],  # C00 += P4, C10 += P4
            },
            # P5 = (A00+A01)*B11
            {
                'a': [(0, 1), (1, 1)],
                'b': [(3, 1)],
                'c': [(0, -1), (1, 1)],  # C00 -= P5, C01 += P5
            },
            # P6 = (A10-A00)*(B00+B01)
            {
                'a': [(2, 1), (0, -1)],
                'b': [(0, 1), (1, 1)],
                'c': [(3, 1)],  # C11 += P6
            },
            # P7 = (A01-A11)*(B10+B11)
            {
                'a': [(1, 1), (3, -1)],
                'b': [(2, 1), (3, 1)],
                'c': [(0, 1)],  # C00 += P7
            },
        ]
        
        # For each block product, for each of the 7 Strassen 2x2 terms:
        for bp in strassen_blocks:
            a_blocks = bp['a']  # list of (block_id, sign)
            b_blocks = bp['b']
            c_blocks = bp['c']
            
            for (su, sv, sw) in strassen_2x2:
                # su, sv, sw are lists of 4 values for the 2x2 local indices
                # Local index 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # For U: combine contributions from all A-blocks
                for (bid, bsign) in a_blocks:
                    for local_idx in range(4):
                        if su[local_idx] != 0:
                            # Determine global A index from block_id and local_idx
                            # local_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                            li, lj = divmod(local_idx, 2)
                            # Block 0: i_base=0, j_base=0
                            # Block 1: i_base=0, j_base=2
                            # Block 2: i_base=2, j_base=0
                            # Block 3: i_base=2, j_base=2
                            i_base = 0 if bid in (0, 1) else 2
                            j_base = 0 if bid in (0, 2) else 2
                            gi = i_base + li
                            gj = j_base + lj
                            gidx = gi * 4 + gj  # U index: i*m + j
                            val = bsign * su[local_idx]
                            u_dict[gidx] = u_dict.get(gidx, 0) + val
                
                # For V: combine contributions from all B-blocks
                # V index: j*p + k, but within 4x4 block, k is local
                # We'll handle the k offset when adding to the global V
                for (bid, bsign) in b_blocks:
                    for local_idx in range(4):
                        if sv[local_idx] != 0:
                            li, lj = divmod(local_idx, 2)
                            i_base = 0 if bid in (0, 1) else 2  # j_base for B
                            j_base = 0 if bid in (0, 2) else 2  # k_base for B
                            gj = i_base + li
                            gk_local = j_base + lj
                            val = bsign * sv[local_idx]
                            # Store with local k, will offset later
                            key = (gj, gk_local)
                            v_dict[key] = v_dict.get(key, 0) + val
                
                # For W: combine contributions from all C-blocks
                # W index: k*n + i = k*4 + i
                for (bid, bsign) in c_blocks:
                    for local_idx in range(4):
                        if sw[local_idx] != 0:
                            li, lj = divmod(local_idx, 2)
                            i_base = 0 if bid in (0, 1) else 2
                            j_base = 0 if bid in (0, 2) else 2
                            gk_local = j_base + lj
                            gi = i_base + li
                            val = bsign * sw[local_idx]
                            key = (gk_local, gi)
                            w_dict[key] = w_dict.get(key, 0) + val
                
                terms.append((u_dict, v_dict, w_dict))
        
        return terms
    
    # Generate the 49 terms for one 4x4 block
    strassen_4x4_terms = get_4x4_strassen_terms()
    
    # Now build the full decomposition
    rank = 49 + 49 + 16  # 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    col = 0
    
    # Add two 4x4 Strassen blocks for B columns 0-3 and 4-7
    for block_col_offset in [0, 4]:
        for (u_dict, v_dict, w_dict) in strassen_4x4_terms:
            # U: global index i*4+j, same as in the 4x4 block
            for (idx, val) in u_dict.items():
                U[idx][col] += val
            
            # V: global index j*9 + k, where k = block_col_offset + k_local
            for ((j, k_local), val) in v_dict.items():
                global_k = block_col_offset + k_local
                v_idx = j * 9 + global_k
                V[v_idx][col] += val
            
            # W: global index k*4 + i, where k = block_col_offset + k_local... 
            # Wait, W index is k*n + i where k is the column index of C
            # C has p=9 columns, so k ranges 0..8
            # For this block, k_local is 0..3, so global k = block_col_offset + k_local
            for ((k_local, i), val) in w_dict.items():
                global_k = block_col_offset + k_local
                w_idx = global_k * 4 + i
                W[w_idx][col] += val
            
            col += 1
    
    # Add schoolbook for the 4x4x1 block (B column 8)
    for i in range(4):
        for j in range(4):
            # C[i, 8] += A[i,j] * B[j, 8]
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
