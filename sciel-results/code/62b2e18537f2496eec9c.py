"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
Strategy: Hybrid decomposition
- 4x4x4 block: Strassen algorithm (rank 7 per 2x2x2 base) -> 7 * 7 = 49 terms
- 4x4x3 block: Schoolbook -> 4*4*3 = 48 terms
Total rank: 97
"""

import time
import itertools

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2x2.
    Each term: (u_terms, v_terms, w_terms)
    where each *_terms is a dict mapping (i,j) to coefficient.
    A entries: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    B entries: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    C entries: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    """
    # M1 = (a+d)(e+h)
    # A side: a at (0,0), d at (1,1)
    # B side: e at (0,0), h at (1,1)
    M1_A = {(0,0): 1, (1,1): 1}
    M1_B = {(0,0): 1, (1,1): 1}
    # (a+d) and (e+h) are 2x2 matrices, their product needs 7 rank-1 terms
    # But wait - in Strassen, M1 is a SINGLE scalar product in the 2x2x2 context
    # No - M1 is a 2x2 matrix product! (a+d) is 2x2, (e+h) is 2x2.
    # Actually no. In Strassen, a,b,c,d,e,f,g,h are SCALARS.
    # So M1 = (a+d)(e+h) is a scalar.
    
    # The 7 Strassen scalars are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 2x2x2 case, each Mj contributes to multiple C entries.
    # We need to decompose each Mj into rank-1 terms where each term
    # is a product of (linear combo of A entries) * (linear combo of B entries) * (indicator for C entry).
    
    # Actually, the standard way: each Mj is ONE scalar. It contributes to specific C entries.
    # So for M1: u has a and d, v has e and h, w has C11 and C22 (with coefficients).
    # But w should have 1 at the C position. Since M1 contributes +1 to C11 and +1 to C22,
    # we need TWO rank-1 terms per Mj? No.
    
    # Let me think more carefully. The tensor is sum_j Mj * (w_pattern_j)
    # where w_pattern_j is a vector with 1s at positions where Mj appears in C.
    # But each term must be rank-1: u_j ⊗ v_j ⊗ w_j where w_j is a vector.
    # w_j can have multiple nonzero entries!
    # So w_j = (1 at C11, 1 at C22) is valid for a rank-1 term.
    
    # M1: A-side: a+d, B-side: e+h, C-side: C11 + C22
    # M2: A-side: c+d, B-side: e,     C-side: C21
    # M3: A-side: a,     B-side: f-h, C-side: C12
    # M4: A-side: d,     B-side: g-e, C-side: C11 + C21
    # M5: A-side: a+b,   B-side: h,   C-side: -C11 + C12
    # M6: A-side: c-a,   B-side: e+f, C-side: C22
    # M7: A-side: b-d,   B-side: g+h, C-side: C11 + C22
    
    # Each is a valid rank-1 term! So 7 terms total.
    
    terms = []
    # (A_dict, B_dict, C_dict)
    # A: {(i,j): coeff} for 2x2 positions
    # B: {(i,j): coeff}
    # C: {(i,j): coeff}
    
    # M1
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
    # M3
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}))
    # M4
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
    # M5
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    # M6
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    
    return terms


def construct():
    """
    Construct rank-97 decomposition for 4x4 * 4x7.
    U: shape (n*m, rank) = (16, 97)
    V: shape (m*p, rank) = (28, 97)
    W: shape (p*n, rank) = (28, 97)
    W index convention: W[k*n + i, r]
    """
    n, m, p = 4, 4, 7
    n_m = n * m  # 16
    m_p = m * p  # 28
    p_n = p * n  # 28
    
    U = [[0.0] * 97 for _ in range(n_m)]
    V = [[0.0] * 97 for _ in range(m_p)]
    W = [[0.0] * 97 for _ in range(p_n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 block (columns 0-3 of output) ---
    # Split into 2x2 blocks
    # A blocks (in A, row-major, A[i,j]):
    #   A11: i∈{0,1}, j∈{0,1}
    #   A12: i∈{0,1}, j∈{2,3}
    #   A21: i∈{2,3}, j∈{0,1}
    #   A22: i∈{2,3}, j∈{2,3}
    
    # B blocks (in B, row-major, B[j,k]):
    #   B11: j∈{0,1}, k∈{0,1}
    #   B12: j∈{0,1}, k∈{2,3}
    #   B21: j∈{2,3}, k∈{0,1}
    #   B22: j∈{2,3}, k∈{2,3}
    
    # C blocks (in C, C[i,k], stored as W[k*n+i]):
    #   C11: i∈{0,1}, k∈{0,1}
    #   C12: i∈{0,1}, k∈{2,3}
    #   C21: i∈{2,3}, k∈{0,1}
    #   C22: i∈{2,3}, k∈{2,3}
    
    # Map 2x2 positions to global indices
    # For A: block (br, bc) with local (lr, lc) -> global row br*2+lr, global col bc*2+lc
    # A11 local (lr,lc) -> A[br*2+lr, bc*2+lc] = A[lr, lc]  (since br=0,bc=0)
    # A12 local (lr,lc) -> A[lr, 2+lc]
    # A21 local (lr,lc) -> A[2+lr, lc]
    # A22 local (lr,lc) -> A[2+lr, 2+lc]
    
    # A index: A[i,j] -> U[i*m + j] = U[i*4+j]
    # B index: B[j,k] -> V[j*p + k] = V[j*7+k]
    # C index: C[i,k] -> W[k*n + i] = W[k*4+i]
    
    # Strassen at the 2x2 block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2*2x2 product. Expand each using the 7 Strassen terms for 2x2x2.
    # This gives 7*7 = 49 terms.
    
    # Define the 7 "meta" Strassen terms
    meta_terms = [
        # (A_comb, B_comb, C_comb)
        # Each comb is: dict from block_name to list of (local_pos, coeff)
        # Actually, let's be more explicit.
        # For each meta term, specify:
        #   which A blocks are involved and how (linear combo of 2x2 matrices)
        #   which B blocks are involved
        #   how the result maps to C blocks
        
        # M1: A11+A22 times B11+B22, contributes to C11 and C22
        {'A': {'A11': [( (0,0),(1,1)) ],  # all 4 local positions with coeff 1
                      'A12': [],
                      'A21': [],
                      'A22': [( (0,0),(1,1)) ]},
         'B': {'B11': [( (0,0),(1,1)) ],
               'B12': [],
               'B21': [],
               'B22': [( (0,0),(1,1)) ]},
         'C': {'C11': [( (0,0),(1,1)) ],
               'C12': [],
               'C21': [],
               'C22': [( (0,0),(1,1)) ]}},
        # M2: (A21+A22) * B11, contributes to C21
        {'A': {'A11': [],
               'A12': [],
               'A21': [( (0,0),(1,1)) ],
               'A22': [( (0,0),(1,1)) ]},
         'B': {'B11': [( (0,0),(1,1)) ],
               'B12': [],
               'B21': [],
               'B22': []},
         'C': {'C11': [],
               'C12': [],
               'C21': [( (0,0),(1,1)) ],
               'C22': []}},
    ]
    
    # This is getting too complicated. Let me use a cleaner approach.
    
    # Each meta term Mj involves a linear combination of A blocks and B blocks.
    # The result is a 2x2 matrix that gets added to specific C blocks.
    # To decompose Mj as a 2x2*2x2 product, I apply the 7 Strassen terms.
    
    # Let me define the 7 meta operations differently.
    # For meta term t, I specify:
    #   A_coeffs: for each of the 4 A-blocks (A11,A12,A21,A22), a 2x2 matrix of coefficients
    #   B_coeffs: same for B-blocks
    #   C_coeffs: for each of the 4 C-blocks, a 2x2 matrix of coefficients (how M contributes)
    
    # Then for each of the 7 Strassen sub-terms within M, I combine.
    
    # Actually, the cleanest way: 
    # For the 2x2 block level Strassen, I have 7 operations.
    # Each operation is: (linear combo of A 2x2-blocks) @ (linear combo of B 2x2-blocks) -> added to C 2x2-blocks.
    # I'll denote each "linear combo of 2x2 blocks" as a 4x4 matrix (stacked blocks).
    
    # Let me just directly enumerate the 49 rank-1 terms.
    
    # For a 2x2 block multiplication, the Strassen terms (from above) are:
    # s1: A=(a,d pattern), B=(e,h pattern), C=(c11,c22 pattern)  
    # where a,b,c,d,e,f,g,h are the scalar entries of the 2x2 inputs.
    
    # In our case, for meta-term Mj, the "2x2 input on A side" has entries that are
    # themselves linear combinations of the original A matrix entries.
    
    # Let me define the 49 terms by combining meta-term and sub-term.
    
    # Meta-term definitions (A-side linear combo as 4x4 matrix over A entries,
    # B-side as 4x4 over B entries, C-side as 4x4 over C entries):
    
    # I'll use the block positions:
    # A11 at (0:2, 0:2), A12 at (0:2, 2:4), A21 at (2:4, 0:2), A22 at (2:4, 2:4)
    # Same for B and C.
    
    def block_matrix(block_dict):
        """Create 4x4 matrix from {block_name: [(lr,lc),...]} where each entry has coeff 1
        or a proper 2x2 matrix."""
        # This won't work generally. Let me be more explicit.
        pass
    
    # OK let me just hard-code the 49 terms directly.
    # 
    # For meta-term Mj, I write out the A-side 2x2 "matrix" where each entry is a 
    # linear combination of A[i,j]. Similarly for B and C.
    #
    # Then apply the 7 Strassen sub-patterns.
    
    # Let me define each meta-term's A, B, C sides as 2x2 arrays of linear forms.
    # A_side: 2x2 array, each entry is a list of (A_index, coeff)
    # where A_index is (i,j) in the 4x4 A matrix.
    
    # A11 local position (lr,lc) corresponds to A[lr, lc]
    # A12 local (lr,lc) -> A[lr, 2+lc]
    # A21 local (lr,lc) -> A[2+lr, lc]
    # A22 local (lr,lc) -> A[2+lr, 2+lc]
    
    # B11 local (lr,lc) -> B[lr, lc]
    # B12 local (lr,lc) -> B[lr, 2+lc]
    # B21 local (lr,lc) -> B[2+lr, lc]
    # B22 local (lr,lc) -> B[2+lr, 2+lc]
    
    # C11 local (lr,lc) -> C[lr, lc] -> W[lc*4+lr]
    # C12 local (lr,lc) -> C[lr, 2+lc] -> W[(2+lc)*4+lr]
    # C21 local (lr,lc) -> C[2+lr, lc] -> W[lc*4+(2+lr)]
    # C22 local (lr,lc) -> C[2+lr, 2+lc] -> W[(2+lc)*4+(2+lr)]
    
    def a11(lr, lc): return (lr, lc)
    def a12(lr, lc): return (lr, 2+lc)
    def a21(lr, lc): return (2+lr, lc)
    def a22(lr, lc): return (2+lr, 2+lc)
    
    def b11(lr, lc): return (lr, lc)
    def b12(lr, lc): return (lr, 2+lc)
    def b21(lr, lc): return (2+lr, lc)
    def b22(lr, lc): return (2+lr, 2+lc)
    
    def c11_lr(lr, lc):  # return (k, i) for W[k*n+i]
        i, k = lr, lc
        return (k, i)
    def c12_lr(lr, lc):
        i, k = lr, 2+lc
        return (k, i)
    def c21_lr(lr, lc):
        i, k = 2+lr, lc
        return (k, i)
    def c22_lr(lr, lc):
        i, k = 2+lr, 2+lc
        return (k, i)
    
    # The 7 meta terms, each specified as:
    # (A_2x2, B_2x2, C_2x2) where each is a 2x2 grid
    # A_2x2[lr][lc] = list of (a_index, coeff)
    # B_2x2[lr][lc] = list of (b_index, coeff)  
    # C_2x2[lr][lc] = list of (c_index_in_W, coeff)
    
    # M1 = (A11 + A22) @ (B11 + B22)
    # A side: [[A11+A22 at (0,0)], ... wait
    # (A11+A22) is a 2x2 matrix where entry (lr,lc) = A11[lr,lc] + A22[lr,lc]
    # = A[lr,lc] + A[2+lr, 2+lc]
    
    def make_add_blocks(blocks):
        """blocks is a list of block functions, returns 2x2 grid of (index, coeff) lists"""
        grid = [[[] for _ in range(2)] for _ in range(2)]
        for func in blocks:
            for lr in range(2):
                for lc in range(2):
                    idx = func(lr, lc)
                    grid[lr][lc].append((idx, 1))
        return grid
    
    def make_diff_blocks(blocks):
        """blocks is list of (block_func, coeff)"""
        grid = [[[] for _ in range(2)] for _ in range(2)]
        for func, c in blocks:
            for lr in range(2):
                for lc in range(2):
                    idx = func(lr, lc)
                    grid[lr][lc].append((idx, c))
        return grid
    
    # M1: A = A11+A22, B = B11+B22, C = C11+C22
    A_M1 = make_add_blocks([a11, a22])
    B_M1 = make_add_blocks([b11, b22])
    C_M1 = [[[(c11_lr(lr,lc), 1), (c22_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M2: A = A21+A22, B = B11, C = C21
    A_M2 = make_add_blocks([a21, a22])
    B_M2 = make_add_blocks([b11])
    C_M2 = [[[(c21_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M3: A = A11, B = B12-B22, C = C12
    A_M3 = make_add_blocks([a11])
    B_M3 = make_diff_blocks([(b12, 1), (b22, -1)])
    C_M3 = [[[(c12_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M4: A = A22, B = B21-B11, C = C11+C21
    A_M4 = make_add_blocks([a22])
    B_M4 = make_diff_blocks([(b21, 1), (b11, -1)])
    C_M4 = [[[(c11_lr(lr,lc), 1), (c21_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M5: A = A11+A12, B = B22, C = -C11+C12
    A_M5 = make_add_blocks([a11, a12])
    B_M5 = make_add_blocks([b22])
    C_M5 = [[[(c11_lr(lr,lc), -1), (c12_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M6: A = A21-A11, B = B11+B12, C = C22
    A_M6 = make_diff_blocks([(a21, 1), (a11, -1)])
    B_M6 = make_add_blocks([b11, b12])
    C_M6 = [[[(c22_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    # M7: A = A12-A22, B = B21+B22, C = C11+C22
    A_M7 = make_diff_blocks([(a12, 1), (a22, -1)])
    B_M7 = make_add_blocks([b21, b22])
    C_M7 = [[[(c11_lr(lr,lc), 1), (c22_lr(lr,lc), 1)] for lc in range(2)] for lr in range(2)]
    
    meta_list = [
        (A_M1, B_M1, C_M1),
        (A_M2, B_M2, C_M2),
        (A_M3, B_M3, C_M3),
        (A_M4, B_M4, C_M4),
        (A_M5, B_M5, C_M5),
        (A_M6, B_M6, C_M6),
        (A_M7, B_M7, C_M7),
    ]
    
    # The 7 Strassen sub-patterns for 2x2x2:
    # Each specified as (a_side_2x2, b_side_2x2, c_side_2x2)
    # where a_side[lr][lc] is coeff for the A-entry at (lr,lc) of the 2x2 input
    # etc.
    
    strassen_sub = [
        # s1: (a+d)(e+h) -> C11+C22
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2: (c+d)*e -> C21
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # s3: a*(f-h) -> C12
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # s4: d*(g-e) -> C11+C21
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5: (a+b)*h -> -C11+C12
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6: (c-a)*(e+f) -> C22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7: (b-d)*(g+h) -> C11+C22
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # For each meta term and each sub-term, generate one rank-1 term.
    # The A-side of the rank-1 term: for each (lr,lc) in sub_pattern, 
    #   for each (a_idx, a_coeff) in meta_A[lr][lc]:
    #     U[a_idx[0]*4 + a_idx[1]] += sub_coeff * a_coeff
    # Similarly for V and W.
    
    for (A_meta, B_meta, C_meta) in meta_list:
        for (a_sub, b_sub, c_sub) in strassen_sub:
            # U vector
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in a_sub:
                        sub_c = a_sub[(lr, lc)]
                        for (a_idx, a_c) in A_meta[lr][lc]:
                            if a_c != 0 and sub_c != 0:
                                a_flat = a_idx[0] * 4 + a_idx[1]
                                U[a_flat][col] += sub_c * a_c
            # V vector
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in b_sub:
                        sub_c = b_sub[(lr, lc)]
                        for (b_idx, b_c) in B_meta[lr][lc]:
                            if b_c != 0 and sub_c != 0:
                                b_flat = b_idx[0] * 7 + b_idx[1]
                                V[b_flat][col] += sub_c * b_c
            # W vector
            for lr in range(2):
                for lc in range(2):
                    if (lr, lc) in c_sub:
                        sub_c = c_sub[(lr, lc)]
                        for (c_idx, c_c) in C_meta[lr][lc]:
                            if c_c != 0 and sub_c != 0:
                                w_k, w_i = c_idx
                                w_flat = w_k * 4 + w_i
                                W[w_flat][col] += sub_c * c_c
            col += 1
    
    # --- Part 2: 4x4x3 block (columns 4-6 of output) ---
    # Schoolbook: 4*4*3 = 48 terms
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4 + j][col] = 1.0
                V[j*7 + (4+k)][col] = 1.0
                W[(4+k)*4 + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
