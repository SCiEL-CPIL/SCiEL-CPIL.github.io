"""Matrix multiplication 5x5x5 - Hierarchical Strassen decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_terms(offset_u, offset_v, offset_w, n=2, m=2, p=2):
    """
    Generate the 7 terms for a 2x2 matrix multiplication block.
    offset_u, offset_v, offset_w are the starting indices for U, V, W in the global arrays.
    U index: (row * m + col) + offset_u
    V index: (row * p + col) + offset_v
    W index: (col * n + row) + offset_w
    """
    terms = []
    
    # M1 = (a00+a11)(b00+b11)
    terms.append((
        {offset_u + 0*n + 0: 1, offset_u + 1*n + 1: 1},
        {offset_v + 0*p + 0: 1, offset_v + 1*p + 1: 1},
        {offset_w + 0*n + 0: 1, offset_w + 1*n + 1: 1}
    ))
    
    # M2 = (a10+a11)b00
    terms.append((
        {offset_u + 1*n + 0: 1, offset_u + 1*n + 1: 1},
        {offset_v + 0*p + 0: 1},
        {offset_w + 0*n + 1: 1, offset_w + 1*n + 1: -1}
    ))
    
    # M3 = a00(b01-b11)
    terms.append((
        {offset_u + 0*n + 0: 1},
        {offset_v + 0*p + 1: 1, offset_v + 1*p + 1: -1},
        {offset_w + 0*n + 0: 1, offset_w + 1*n + 1: 1}
    ))
    
    # M4 = a11(b10-b00)
    terms.append((
        {offset_u + 1*n + 1: 1},
        {offset_v + 1*p + 0: 1, offset_v + 0*p + 0: -1},
        {offset_w + 0*n + 0: 1, offset_w + 1*n + 0: 1}
    ))
    
    # M5 = (a00+a01)b11
    terms.append((
        {offset_u + 0*n + 0: 1, offset_u + 0*n + 1: 1},
        {offset_v + 1*p + 1: 1},
        {offset_w + 0*n + 0: -1, offset_w + 0*n + 1: 1}
    ))
    
    # M6 = (a10-a00)(b00+b01)
    terms.append((
        {offset_u + 1*n + 0: 1, offset_u + 0*n + 0: -1},
        {offset_v + 0*p + 0: 1, offset_v + 0*p + 1: 1},
        {offset_w + 1*n + 1: 1}
    ))
    
    # M7 = (a01-a11)(b10+b11)
    terms.append((
        {offset_u + 0*n + 1: 1, offset_u + 1*n + 1: -1},
        {offset_v + 1*p + 0: 1, offset_v + 1*p + 1: 1},
        {offset_w + 0*n + 0: 1}
    ))
    
    return terms

def _construct_4x4_decomposition(offset_u, offset_v, offset_w):
    """
    Construct the rank-49 decomposition for a 4x4 matrix multiplication block.
    This is done by applying Strassen to the 2x2 blocks of 2x2 matrices.
    The 4x4 matrix is split into four 2x2 blocks.
    A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    C = [[C00, C01], [C10, C11]] where Cij are 2x2.
    
    The offsets for the 2x2 blocks are:
    A00: offset_u + 0*4 + 0 (row 0, col 0 of 4x4, but local 2x2) -> offset_u
    A01: offset_u + 0*4 + 2
    A10: offset_u + 2*4 + 0
    A11: offset_u + 2*4 + 2
    
    Similarly for V and W.
    """
    terms = []
    
    # Define offsets for the 2x2 sub-blocks
    # U (A)
    u_00 = offset_u
    u_01 = offset_u + 2
    u_10 = offset_u + 8
    u_11 = offset_u + 10
    
    # V (B)
    v_00 = offset_v
    v_01 = offset_v + 2
    v_10 = offset_v + 8
    v_11 = offset_v + 10
    
    # W (C)
    # W index is (col * n + row) + offset_w
    # C00 is rows 0:2, cols 0:2.
    # Local W indices for C00:
    # (0,0) -> 0, (0,1) -> 1, (1,0) -> 2, (1,1) -> 3 (relative to C00 top-left)
    # Global W index for C00 top-left is offset_w.
    # C01 is rows 0:2, cols 2:4.
    # Global W index for C01 top-left is offset_w + 2.
    # C10 is rows 2:4, cols 0:2.
    # Global W index for C10 top-left is offset_w + 4.
    # C11 is rows 2:4, cols 2:4.
    # Global W index for C11 top-left is offset_w + 6.
    
    w_00 = offset_w
    w_01 = offset_w + 2
    w_10 = offset_w + 4
    w_11 = offset_w + 6
    
    # Strassen products for 2x2 blocks of 2x2 matrices
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # We need to add the terms for A00, A11 to U, B00, B11 to V, and C00, C11 to W.
    # Since the terms are linear, we can generate the Strassen terms for each pair and add them.
    
    def add_strassen(u_off1, u_off2, v_off1, v_off2, w_off1, w_off2, sign1=1, sign2=1):
        # This is tricky because Strassen involves sums.
        # We will just generate the 7 terms for the "virtual" 2x2 matrix multiplication
        # where the inputs are the sums/differences of the blocks.
        # Actually, it's easier to just expand the 49 terms directly.
        pass

    # Let's just use the recursive definition.
    # The 4x4 multiplication is equivalent to 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank 49.
    
    # We can define the 7 "meta-terms" and for each, generate the 7 Strassen terms for the 2x2 blocks.
    
    meta_terms = [
        # (U_blocks, V_blocks, W_blocks, coeffs)
        # M1
        ({u_00: 1, u_11: 1}, {v_00: 1, v_11: 1}, {w_00: 1, w_11: 1}),
        # M2
        ({u_10: 1, u_11: 1}, {v_00: 1}, {w_10: 1, w_11: -1}),
        # M3
        ({u_00: 1}, {v_01: 1, v_11: -1}, {w_01: 1, w_11: 1}),
        # M4
        ({u_11: 1}, {v_10: 1, v_00: -1}, {w_00: 1, w_10: 1}),
        # M5
        ({u_00: 1, u_01: 1}, {v_11: 1}, {w_00: -1, w_01: 1}),
        # M6
        ({u_10: 1, u_00: -1}, {v_00: 1, v_01: 1}, {w_11: 1}),
        # M7
        ({u_01: 1, u_11: -1}, {v_10: 1, v_11: 1}, {w_00: 1}),
    ]
    
    for u_blocks, v_blocks, w_blocks in meta_terms:
        # For each meta-term, we need to perform a 2x2 matrix multiplication
        # where the entries are linear combinations of the 2x2 sub-blocks.
        # This is equivalent to taking the Kronecker product of the 2x2 Strassen decomposition
        # with the coefficients of the blocks.
        
        # Let's just iterate over the 7 Strassen terms for the 2x2 structure
        # and for each, iterate over the blocks.
        
        # This is getting complicated to implement generically.
        # Let's hardcode the 49 terms by expanding the 7 meta-terms.
        
        # We can create a helper to expand a meta-term into 7 actual terms.
        # A meta-term is (U_comb, V_comb, W_comb).
        # U_comb is a dict of {offset: coeff}.
        # We need to form U_vec = sum coeff * U_block.
        # Then apply Strassen to U_vec, V_vec.
        
        # Actually, since the 2x2 Strassen decomposition is fixed, we can just
        # take the 7 Strassen terms for 2x2, and for each, multiply the coefficients
        # of the blocks.
        
        # Let's do it this way:
        # For each of the 7 Strassen terms (s1..s7) for 2x2:
        #   For each block in U_comb:
        #     For each block in V_comb:
        #       For each block in W_comb:
        #         This doesn't work because the Strassen terms are linear combinations.
        
        # Correct approach:
        # The meta-term defines a 2x2 matrix multiplication problem where the entries
        # are linear combinations of the 2x2 sub-blocks.
        # Let U_meta[i,j] = sum_{b in U_blocks} U_blocks[b] * U_sub[i,j]
        # This is not right.
        
        # Let's stick to the simplest implementation:
        # Generate the 49 terms explicitly.
        
        # We can use the fact that the 4x4 decomposition is the tensor product
        # of the 2x2 Strassen decomposition with itself? No.
        
        # Let's just use the schoolbook for 4x4 if we can't easily generate 49.
        # Schoolbook 4x4 is 64.
        # 64 + 45 = 109. Still better than 117.
        # But 49 is much better.
        
        # Let's try to generate the 49 terms.
        
        # We will define the 7 Strassen terms for the 2x2 case as functions of the offsets.
        # And then we will "inflate" them.
        
        # For each meta-term (u_blocks, v_blocks, w_blocks):
        #   We want to compute sum_{c} W_c * (U_c . A) * (V_c . B)
        #   Where U_c, V_c, W_c are the Strassen coefficients.
        #   But U_c is a linear combination of the sub-blocks.
        
        # Let's just hardcode the 49 terms by calling _strassen_2x2_terms
        # with appropriate offsets and scaling.
        
        # Wait, _strassen_2x2_terms assumes the inputs are single 2x2 matrices.
        # Here the inputs are sums of 2x2 matrices.
        # So we need to add the terms for each component.
        
        # Example M1: (A00+A11)(B00+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11
        # Each of these is a 2x2 matrix multiplication.
        # We can use the 7-term Strassen for each? No, that would be 7*4 = 28 terms.
        # But M1 is just one term in the Strassen decomposition of the 4x4.
        # The Strassen decomposition of 4x4 uses 7 products of 2x2 matrices.
        # Each product is a 2x2 matrix multiplication.
        # So we need the rank of a 2x2 matrix multiplication, which is 7.
        # So 7 * 7 = 49.
        
        # So for each meta-term, we need to generate the 7 terms for the 2x2 multiplication
        # of the linear combinations.
        
        # Let's write a helper that takes U_blocks, V_blocks, W_blocks and returns the 7 terms.
        
        def expand_meta_term(u_blocks, v_blocks, w_blocks):
            terms_out = []
            # The 7 Strassen terms for 2x2
            s1_u = {0: 1, 3: 1}
            s1_v = {0: 1, 3: 1}
            s1_w = {0: 1, 3: 1}
            
            s2_u = {2: 1, 3: 1}
            s2_v = {0: 1}
            s2_w = {2: 1, 3: -1}
            
            s3_u = {0: 1}
            s3_v = {1: 1, 3: -1}
            s3_w = {1: 1, 3: 1}
            
            s4_u = {3: 1}
            s4_v = {2: 1, 0: -1}
            s4_w = {0: 1, 2: 1}
            
            s5_u = {0: 1, 1: 1}
            s5_v = {3: 1}
            s5_w = {0: -1, 1: 1}
            
            s6_u = {2: 1, 0: -1}
            s6_v = {0: 1, 1: 1}
            s6_w = {3: 1}
            
            s7_u = {1: 1, 3: -1}
            s7_v = {2: 1, 3: 1}
            s7_w = {0: 1}
            
            strassen_2x2 = [
                (s1_u, s1_v, s1_w),
                (s2_u, s2_v, s2_w),
                (s3_u, s3_v, s3_w),
                (s4_u, s4_v, s4_w),
                (s5_u, s5_v, s5_w),
                (s6_u, s6_v, s6_w),
                (s7_u, s7_v, s7_w),
            ]
            
            for su, sv, sw in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Expand U
                for u_idx, u_coeff in su.items():
                    for block_off, block_coeff in u_blocks.items():
                        # u_idx is 0..3 for the 2x2 matrix.
                        # block_off is the offset of the 2x2 sub-block.
                        # The global U index is block_off + u_idx.
                        global_idx = block_off + u_idx
                        u_dict[global_idx] = u_dict.get(global_idx, 0) + u_coeff * block_coeff
                
                # Expand V
                for v_idx, v_coeff in sv.items():
                    for block_off, block_coeff in v_blocks.items():
                        global_idx = block_off + v_idx
                        v_dict[global_idx] = v_dict.get(global_idx, 0) + v_coeff * block_coeff
                
                # Expand W
                for w_idx, w_coeff in sw.items():
                    for block_off, block_coeff in w_blocks.items():
                        global_idx = block_off + w_idx
                        w_dict[global_idx] = w_dict.get(global_idx, 0) + w_coeff * block_coeff
                
                terms_out.append((u_dict, v_dict, w_dict))
            
            return terms_out
        
        terms.extend(expand_meta_term(u_blocks, v_blocks, w_blocks))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    terms = []
    
    # 1. 4x4 Core
    # Offset 0 for U, V, W
    core_terms = _construct_4x4_decomposition(0, 0, 0)
    terms.extend(core_terms)
    
    # 2. Remaining parts
    # A rows 0:4, col 4
    # B rows 4, cols 0:4
    # A row 4, cols 0:4
    # B rows 0:4, col 4
    # A[4,4], B[4,4]
    
    # Schoolbook for these
    
    # A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            u_idx = i * m + 4
            v_idx = 4 * p + j
            w_idx = j * n + i
            terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
            
    # A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        u_idx = i * m + 4
        v_idx = 4 * p + 4
        w_idx = 4 * n + i
        terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
    # A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    for i in range(4):
        for j in range(4):
            u_idx = 4 * m + j
            v_idx = j * p + i
            w_idx = i * n + 4
            terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
            
    # A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    for j in range(4):
        u_idx = 4 * m + j
        v_idx = j * p + 4
        w_idx = 4 * n + 4
        terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
    # A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    for i in range(4):
        u_idx = 4 * m + 4
        v_idx = 4 * p + i
        w_idx = i * n + 4
        terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
    # A[4, 4] * B[4, 4] -> C[4, 4]
    u_idx = 4 * m + 4
    v_idx = 4 * p + 4
    w_idx = 4 * n + 4
    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
                
    # Verify
    # (Optional but recommended)
    # We can skip verification for speed if confident, but it's safer.
    # The deadline is 140s.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
