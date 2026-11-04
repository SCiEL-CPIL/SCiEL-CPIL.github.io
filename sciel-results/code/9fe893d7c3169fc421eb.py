import time
import numpy as np

DIMENSIONS = [4, 4, 9]
N, M, P = 4, 4, 9
DIM_U = N * M  # 16
DIM_V = M * P  # 36
DIM_W = P * N  # 36

# Strassen 2x2x2 terms: (u, v, w) where each is a 4-vector for 2x2 block
# Index order: 0=00, 1=01, 2=10, 3=11
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def _strassen_444_terms():
    """Generate 49 rank-1 terms for 4x4x4 Strassen decomposition.
    Returns list of (u16, v16, w16) where each is a 16-vector for the 4x4 block."""
    # 4x4 split into four 2x2 blocks:
    # Block indices: 0=top-left(00), 1=top-right(01), 2=bottom-left(10), 3=bottom-right(11)
    # Within each 2x2 block, element order: 0=00,1=01,2=10,3=11
    # Global 4x4 index (row-major): block*4 + local_idx
    
    terms = []
    for (u, v, w) in _STRASSEN_222:
        # Build 16-vectors for U, V, W
        u16 = [0] * 16
        v16 = [0] * 16
        w16 = [0] * 16
        
        # Map Strassen 2x2 block coefficients to 4x4
        # For each of the 7 Strassen multiplications, the 2x2 blocks involved
        # The 4x4 = 4 blocks of 2x2, so each Strassen term contributes to specific 2x2 sub-blocks
        
        # Actually, let me think differently. The 4x4x4 Strassen uses 7 recursive 2x2x2 calls.
        # Standard 4x4 Strassen: split into 2x2 blocks, apply Strassen at that level.
        # The 7 terms at the top level operate on 2x2 blocks.
        
        # Block layout for 4x4 (row-major):
        # A = [[A00, A01],
        #      [A10, A11]]  where each Ai is 2x2
        # Global index for block (r,c) element (i,j): (r*2+i)*4 + (c*2+j)
        
        # For Strassen at the 2x2-block level:
        # m1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        # m2 = (A10+A11)*B00 -> C10
        # m3 = A00*(B01-B11) -> C01
        # m4 = A11*(B10-B00) -> C00, C10
        # m5 = (A00+A01)*B11 -> C01
        # m6 = (A10-A00)*(B00+B01) -> C10
        # m7 = (A01-A11)*(B10+B11) -> C11
        
        # For each Strassen term, we need to expand to 2x2 level
        # But actually, the standard approach is: the 4x4x4 Strassen has rank 7*4=28? No.
        # 4x4x4 Strassen rank = 7^2 = 49? No, that's not right either.
        
        # Let me reconsider. The current code's _add_444 uses 7 terms with 16-vectors.
        # Let me just use the verified construction from the current code.
        pass
    
    # I'll use a direct approach: verify the current code's _add_444 coefficients
    # and replicate them, but with proper global indexing for the 4x4x9 problem.
    
    # From the current code's _add_444:
    U_sums = [
        [1, 0, 0, 1],   # A00 + A11
        [0, 1, 0, 0],   # A01
        [0, 0, 1, 0],   # A10
        [1, 1, 1, 1],   # A00 + A01 + A10 + A11
        [0, 0, 1, 1],   # A10 + A11
        [1, 0, 0, 0],   # A00
        [0, 1, 1, 1],   # A01 + A10 + A11
    ]
    V_sums = [
        [1, 0, 0, 1],   # B00 + B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01 - B11
        [0, -1, 1, 0],  # B10 - B00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00 + B01
        [0, 0, 1, 1],   # B10 + B11
    ]
    W_sums = [
        [1, 0, 0, 1],   # C00 + C11
        [0, 0, 1, 0],   # C10
        [0, 1, 0, 0],   # C01
        [1, 0, 1, 0],   # C00 + C10
        [-1, 1, 0, 0],  # C01 - C00
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0],   # C00
    ]
    
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        u16 = [0] * 16
        v16 = [0] * 16
        w16 = [0] * 16
        for s in range(4):
            # s is the 2x2 block index: 0=00, 1=01, 2=10, 3=11
            # Global 4x4 index for block s: s * 4 (since each block is 2x2=4 elements, row-major)
            # Wait, block 0 covers indices 0-3, block 1 covers 4-7, block 2 covers 8-11, block 3 covers 12-15
            base = s * 4
            u16[base] = u[s]  # This is wrong - u[s] is a scalar for the whole block?
            
            # Actually, looking at the current code more carefully:
            # In _add_444, for each t (0-6) and s (0-3):
            #   if u[s]: U[a_base + s][col] += u[s]
            # This means U_sums[t][s] is added to U at position a_base+s
            # So U_sums[t] is a 4-vector where index s corresponds to... 
            # a_base + s, which are 4 consecutive elements.
            # But a 4x4 block has 16 elements. So this must be using a different indexing.
            
            # I think the current code treats the 4x4 as having 4 "super-rows" of 1 element each?
            # That doesn't make sense. Let me re-read.
            
            # Actually I think there's a bug or I'm misreading. Let me just build the 4x4x4
            # Strassen from scratch using the 2x2x2 Strassen recursively.
        pass
    
    # Let me build 4x4x4 Strassen properly using 2x2x2 blocks
    # A 4x4 matrix can be viewed as 4 2x2 blocks:
    # A = [A00 A01; A10 A11] where Ai is 2x2
    # Global index (i,j) -> i*4+j
    # Block A00 covers (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
    # Block A01 covers (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
    # Block A10 covers (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
    # Block A11 covers (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
    
    # For Strassen at the 4x4 level, we have 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7 (via Strassen 2x2x2).
    # So total rank = 7 * 7 = 49.
    
    # Let me enumerate all 49 terms.
    # For each of the 7 Strassen multipliers (m1..m7), and for each of the 7 
    # sub-multipliers within that 2x2x2 Strassen, we get one rank-1 term.
    
    # The 7 Strassen multipliers for 4x4x4:
    # Let A = [A00 A01; A10 A11], B = [B00 B01; B10 B11], C = C00 C01; C10 C11
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    
    # And:
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    
    # For each mi, it's a 2x2 matrix product, decomposed via Strassen 2x2x2 into 7 terms.
    
    terms = []
    for m_idx in range(7):
        # Determine which 2x2 blocks are involved in mi
        # and the signs
        if m_idx == 0:  # m1 = (A00+A11)*(B00+B11)
            a_blocks = [(0, 1), (3, 1)]  # (block_idx, sign)
            b_blocks = [(0, 1), (3, 1)]
            # For each pair of a_block, b_block, apply Strassen 2x2x2
            # Actually, (A00+A11) is a sum, so we need to distribute
            # (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
            # That's 4 separate 2x2x2 products, each rank 7, total 28. Not good.
            pass
        
        # This is getting complicated. Let me use a different approach.
        # I'll directly construct the 4x4x4 Strassen by treating it as
        # 4 separate 2x2x2 Strassen calls for the 4 output 2x2 blocks,
        # but that's just schoolbook at the block level (16 terms * 7 = 112).
        
        # The standard 4x4x4 Strassen with rank 49 works by applying
        # Strassen's algorithm recursively. Let me just hardcode the 49 terms.
        
    # Actually, I realize the current code's _add_444 might have an error in my understanding.
    # Let me just use the 4+4+1 split with two verified 4x4x4 Strassen decompositions
    # and one schoolbook column.
    
    # For a clean 4x4x4 Strassen, I'll use the known result that it has rank 49.
    # I'll construct it by doing 7 Strassen 2x2x2 multiplications, each contributing 7 terms.
    
    # Block index mapping for 4x4 (row-major, 16 elements):
    # 2x2 block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Element (i,j) in block (br,bc) has global index i*4+j where i=2*br+di, j=2*bc+dj
    
    def block_elem_idx(br, bc, di, dj):
        i = 2 * br + di
        j = 2 * bc + dj
        return i * 4 + j
    
    # For each of the 7 top-level Strassen multipliers, and for each of the 7 
    # sub-Strassen terms, build the 16-vectors.
    
    # Top-level Strassen multipliers (which 2x2 blocks they use):
    # m1: A blocks {0,3}, B blocks {0,3}, with signs +1
    # m2: A blocks {2,3}, B blocks {0}, with signs +1
    # m3: A blocks {0}, B blocks {1,3}, with signs +1,-1
    # m4: A blocks {3}, B blocks {2,0}, with signs +1,-1
    # m5: A blocks {0,1}, B blocks {3}, with signs +1
    # m6: A blocks {2,0}, B blocks {0,1}, with signs +1,-1
    # m7: A blocks {1,3}, B blocks {2,3}, with signs +1,-1
    
    # For each multiplier, it's (sum of signed A blocks) * (sum of signed B blocks)
    # = sum of (sign_a * sign_b * A_block * B_block)
    # Each A_block * B_block is a 2x2x2 product, rank 7 via Strassen.
    
    # For m1: (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # That's 4 products, each rank 7 = 28 terms. Total for all 7 multipliers would be > 49.
    
    # I think the issue is that the 4x4x4 Strassen with rank 49 is not simply
    # 7 independent 2x2x2 Strassen calls. It requires a more clever combination.
    
    # Let me just use the schoolbook-free approach: 4x4x4 has a known rank-49 decomposition.
    # I'll look up or derive it. Actually, for the purpose of this task, let me just
    # use the 4+4+1 split and verify that each 4x4x4 part is correct.
    
    # For simplicity and correctness, let me use:
    # - Two 4x4x4 Strassen blocks (if I can verify rank 49 each)
    # - One schoolbook column (rank 16)
    # Total: 49 + 49 + 16 = 114
    
    # But I need to actually construct the 4x4x4 Strassen correctly.
    
    # Alternative: use the current code's _add_444 which claims rank 49.
    # Let me trust it and adapt the indexing.
    
    # From current code _add_444, it uses U_sums[t][s] where s=0..3,
    # and adds to U[a_base + s]. This suggests it's treating the 4x4 block
    # as having 4 "super-indices", which doesn't match a 16-element vector.
    
    # I think the current code's _add_444 is actually incorrect or uses a
    # non-standard indexing. Let me just build everything from scratch
    # using only verified 2x2x2 Strassen blocks.
    
    # Plan: decompose 4x4x9 as:
    #   4x4x4 + 4x4x4 + 4x4x1
    # For each 4x4x4: use schoolbook at the 2x2 block level:
    #   4 block-pairs * 7 (Strassen) = 28 per 4x4x4? No, that's not right.
    
    # For a 4x4x4 product using 2x2x2 Strassen blocks:
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each 2x2 product has rank 7 via Strassen.
    # Total: 8 * 7 = 56 terms for 4x4x4.
    
    # So 4x4x9 = 56 + 56 + 16 (schoolbook for 4x4x1) = 128.
    # Or: 4x4x4 + 4x4x4 + 4x4x1 = 56 + 56 + 16 = 128.
    
    # Current code does 49 + 28 + 48 = 125. My approach gives 128. Worse.
    
    # Let me try: 4x4x4 (56) + 4x4x5 where 4x4x5 = 4x4x4(56) + 4x4x1(16) = 72.
    # Total: 56 + 72 = 128. Same.
    
    # The current code's 125 is better. Let me try to understand its _add_444.
    
    # Actually, I'll just replicate the current code's structure but fix the
    # indexing to be correct. The current code scores 1.0 which means rank 144
    # (schoolbook fallback), so its _add_444 must be failing verification.
    
    # Let me build a correct 4x4x4 using 2x2x2 Strassen blocks (rank 56)
    # and a 4x4x3 using schoolbook (rank 48), plus 4x4x2 using 2x2x2 Strassen (rank 28).
    # Total: 56 + 28 + 48 = 132. Worse than 125.
    
    # Hmm. Let me try a different split: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # 4x4x4 via 2x2x2 Strassen: 56 each
    # 4x4x1 via schoolbook: 16
    # Total: 56 + 56 + 16 = 128
    
    # Or: 4x4x9 = 4x4x8 + 4x4x1
    # 4x4x8 = 2 * 4x4x4 = 2 * 56 = 112
    # 4x4x1 = 16
    # Total: 128
    
    # I can't beat 125 with simple block decompositions. Let me just use
    # the 4+4+1 split with 2x2x2 Strassen for the 4x4x4 parts (56 each)
    # and schoolbook for the last column (16). Total: 128.
    
    # Actually, wait. Let me re-examine the current code's _add_444 more carefully.
    # It says "rank-49 decomposition" but uses only 7 terms with 4-element vectors.
    # 7 terms * (some expansion) = 49? That would require each term to expand to 7.
    
    # I think the current code is buggy. Let me just implement a correct solution
    # with rank 128 and see if that's better than 144.
    
    return []  # placeholder


def _add_strassen_222_block(U, V, W, col, a_base, b_base, c_base):
    """Add 7 rank-1 terms for a 2x2x2 Strassen block.
    a_base, b_base, c_base are global indices of the first element of each 2x2 block.
    The 2x2 block elements are at indices: base, base+1, base+2, base+3 (row-major within 2x2).
    """
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_444_via_222(U, V, W, col, a_base, b_base, c_base):
    """4x4x4 via 2x2x2 Strassen blocks. Rank 56.
    A, B, C are 4x4 blocks. We decompose into 8 2x2x2 products.
    A block layout (row-major 16):
      A00: a_base+0..3, A01: a_base+4..7, A10: a_base+8..11, A11: a_base+12..15
    Wait, that's if we store 2x2 blocks contiguously. But in row-major 4x4:
      Row 0: cols 0,1,2,3 -> A00(0,0)=idx0, A00(0,1)=idx1, A01(0,0)=idx2, A01(0,1)=idx3
      Row 1: A00(1,0)=idx4, A00(1,1)=idx5, A01(1,0)=idx6, A01(1,1)=idx7
      Row 2: A10(0,0)=idx8, A10(0,1)=idx9, A11(0,0)=idx10, A11(0,1)=idx11
      Row 3: A10(1,0)=idx12, A10(1,1)=idx13, A11(1,0)=idx14, A11(1,1)=idx15
    
    So 2x2 block (br,bc) has elements at:
      (2*br+di)*4 + (2*bc+dj) for di,dj in {0,1}
    
    For _add_strassen_222_block, I need 4 consecutive indices. But the 2x2 blocks
    in a 4x4 are NOT stored as 4 consecutive elements. They're interleaved.
    
    I need a different helper that takes explicit indices.
    """
    
    def _block_indices(base, br, bc):
        """Return the 4 global indices for 2x2 block (br,bc) in a 4x4 block at 'base'."""
        return [
            base + (2*br+0)*4 + (2*bc+0),
            base + (2*br+0)*4 + (2*bc+1),
            base + (2*br+1)*4 + (2*bc+0),
            base + (2*br+1)*4 + (2*bc+1),
        ]
    
    # 8 2x2x2 products:
    products = [
        # (a_br, a_bc, b_br, b_bc, c_br, c_bc)
        (0, 0, 0, 0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1, 0, 0, 0),  # A01*B10 -> C00
        (0, 0, 0, 1, 0, 1),  # A00*B01 -> C01
        (0, 1, 1, 1, 0, 1),  # A01*B11 -> C01
        (1, 0, 0, 0, 1, 0),  # A10*B00 -> C10
        (1, 1, 1, 0, 1, 0),  # A11*B10 -> C10
        (1, 0, 0, 1, 1, 1),  # A10*B01 -> C11
        (1, 1, 1, 1, 1, 1),  # A11*B11 -> C11
    ]
    
    for (abr, abc, bbr, bbc, cbr, cbc) in products:
        a_idx = _block_indices(a_base, abr, abc)
        b_idx = _block_indices(b_base, bbr, bbc)
        c_idx = _block_indices(c_base, cbr, cbc)
        
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[a_idx[t]][col] += u[t]
                if v[t]:
                    V[b_idx[t]][col] += v[t]
                if w[t]:
                    W[c_idx[t]][col] += w[t]
            col += 1
    
    return col


def _add_442_via_222(U, V, W, col, a_base, b_base, c_base):
    """4x4x2 via 2x2x2 Strassen blocks. Rank 28.
    A is 4x4, B is 4x2, C is 4x2.
    B has two 2x2 blocks: B0 (cols 0-1) and B1 (cols 2-3) if B were 4x4,
    but B is 4x2, so only one column of 2x2 blocks: B_top (rows 0-1) and B_bot (rows 2-3).
    
    Actually B is 4x2: 4 rows, 2 cols. In row-major: row0: b[0,0],b[0,1], row1: b[1,0],b[1,1], etc.
    So B is just a 4x2 matrix. To use 2x2x2 Strassen, I need B to have 2x2 blocks.
    B = [B00; B10] where B00 is rows 0-1 (2x2) and B10 is rows 2-3 (2x2).
    
    A = [A00 A01; A10 A11] (2x2 blocks of 2x2)
    C = [C00; C10] where C00 is rows 0-1, C10 is rows 2-3.
    
    C00 = A00*B00 + A01*B10 (two 2x2x2 products)
    C10 = A10*B00 + A11*B10 (two 2x2x2 products)
    Total: 4 * 7 = 28 terms.
    """
    
    def _a_block(a_base, br, bc):
        return [
            a_base + (2*br+0)*4 + (2*bc+0),
            a_base + (2*br+0)*4 + (2*bc+1),
            a_base + (2*br+1)*4 + (2*bc+0),
            a_base + (2*br+1)*4 + (2*bc+1),
        ]
    
    def _b_block(b_base, br):
        # B is 4x2, block br (0 or 1) covers rows 2*br to 2*br+1
        return [
            b_base + (2*br+0)*2 + 0,
            b_base + (2*br+0)*2 + 1,
            b_base + (2*br+1)*2 + 0,
            b_base + (2*br+1)*2 + 1,
        ]
    
    def _c_block(c_base, br):
        return [
            c_base + (2*br+0)*2 + 0,
            c_base + (2*br+0)*2 + 1,
            c_base + (2*br+1)*2 + 0,
            c_base + (2*br+1)*2 + 1,
        ]
    
    products = [
        # (a_br, a_bc, b_br, c_br)
        (0, 0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1, 0),  # A01*B10 -> C00
        (1, 0, 0, 1),  # A10*B00 -> C10
        (1, 1, 1, 1),  # A11*B10 -> C10
    ]
    
    for (abr, abc, bbr, cbr) in products:
        a_idx = _a_block(a_base, abr, abc)
        b_idx = _b_block(b_base, bbr)
        c_idx = _c_block(c_base, cbr)
        
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[a_idx[t]][col] += u[t]
                if v[t]:
                    V[b_idx[t]][col] += v[t]
                if w[t]:
                    W[c_idx[t]][col] += w[t]
            col += 1
    
    return col


def _schoolbook_4x4xK(U, V, W, col, a_base, b_base, c_base, K):
    """Schoolbook for 4x4 * 4xK -> 4xK. Rank = 16*K."""
    for i in range(4):
        for j in range(4):
            for k in range(K):
                U[a_base + i*4 + j][col] += 1
                V[b_base + j*4 + k][col] += 1  # B is 4xK, row j, col k -> j*K+k... 
                # Wait, B is 4xK: row j has K elements. Index = j*K + k.
                # But b_base is the start of B. If B is stored as 4xK row-major, index = b_base + j*K + k.
                # Hmm, but in the global problem, B is 4x9, so a "4xK sub-block" starting at column k0
                # has elements at b_base + j*9 + (k0+k) for global B.
                # I'll handle this in the caller.
                W[c_base + i*K + k][col] += 1  # C is 4xK: index = i*K + k
                col += 1
    return col


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    
    # Strategy: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # 4x4x4 via 2x2x2 Strassen: rank 56 each
    # 4x4x1 via schoolbook: rank 16
    # Total: 56 + 56 + 16 = 128
    
    rank = 128
    U = [[0] * rank for _ in range(DIM_U)]
    V = [[0] * rank for _ in range(DIM_V)]
    W = [[0] * rank for _ in range(DIM_W)]
    col = 0
    
    # Part 1: 4x4x4 for cols 0-3
    # A: 4x4 at index 0 (16 elements)
    # B: 4x4 sub-block of the 4x9 B, cols 0-3. In global B (4x9, row-major), 
    #    col 0-3 of B means k=0..3. B element (j,k) is at j*9+k.
    #    So the 4x4 sub-block has elements at j*9+k for j=0..3, k=0..3.
    #    These are NOT contiguous. I need to pass explicit indices.
    
    # Let me handle this more carefully by building index lists.
    
    # Global A: 4x4, element (i,j) at i*4+j. 16 contiguous elements starting at 0.
    # Global B: 4x9, element (j,k) at j*9+k.
    # Global C: 4x4 (trace convention), element (k,i) at k*4+i. 16 contiguous starting at 0.
    
    # For Part 1 (cols 0-3):
    # A: all 16 elements, indices 0..15
    # B: elements (j,k) for j=0..3, k=0..3 -> indices j*9+k = 0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30
    # C: elements (k,i) for k=0..3, i=0..3 -> indices k*4+i = 0..15 (contiguous)
    
    # For _add_444_via_222, I need a_base, b_base, c_base such that the 4x4 blocks
    # are stored in a specific layout. But B's 4x4 sub-block is NOT contiguous in the global array.
    
    # I need a more flexible approach: pass explicit index lists for each 2x2 block.
    
    # Let me create a helper that adds a 2x2x2 Strassen term given explicit index lists.
    
    def _add_222_strassen(U, V, W, col, a_idx, b_idx, c_idx):
        """Add 7 rank-1 Strassen terms for 2x2x2 with explicit index lists."""
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[a_idx[t]][col] += u[t]
                if v[t]:
                    V[b_idx[t]][col] += v[t]
                if w[t]:
                    W[c_idx[t]][col] += w[t]
            col += 1
        return col
    
    # Build index lists for Part 1 (cols 0-3)
    # A 2x2 blocks:
    A00 = [0, 1, 4, 5]    # (0,0),(0,1),(1,0),(1,1)
    A01 = [2, 3, 6, 7]    # (0,2),(0,3),(1,2),(1,3)
    A10 = [8, 9, 12, 13]  # (2,0),(2,1),(3,0),(3,1)
    A11 = [10, 11, 14, 15]# (2,2),(2,3),(3,2),(3,3)
    
    # B 2x2 blocks (cols 0-3 of 4x9):
    B00 = [0, 1, 9, 10]   # (0,0),(0,1),(1,0),(1,1) -> 0*9+0, 0*9+1, 1*9+0, 1*9+1
    B01 = [2, 3, 11, 12]  # (0,2),(0,3),(1,2),(1,3)
    B10 = [18, 19, 27, 28]# (2,0),(2,1),(3,0),(3,1) -> 2*9+0, 2*9+1, 3*9+0, 3*9+1
    B11 = [20, 21, 29, 30]# (2,2),(2,3),(3,2),(3,3)
    
    # C 2x2 blocks (cols 0-3 of C, trace convention C[k,i] at k*4+i):
    C00 = [0, 1, 4, 5]    # (k=0,i=0),(0,1),(1,0),(1,1)
    C01 = [2, 3, 6, 7]    # (0,2),(0,3),(1,2),(1,3)
    C10 = [8, 9, 12, 13]  # (2,0),(2,1),(3,0),(3,1)
    C11 = [10, 11, 14, 15]# (2,2),(2,3),(3,2),(3,3)
    
    # 8 products for 4x4x4:
    products = [
        (A00, B00, C00),
        (A01, B10, C00),
        (A00, B01, C01),
        (A01, B11, C01),
        (A10, B00, C10),
        (A11, B10, C10),
        (A10, B01, C11),
        (A11, B11, C11),
    ]
    
    for a_idx, b_idx, c_idx in products:
        col = _add_222_strassen(U, V, W, col, a_idx, b_idx, c_idx)
    
    # Part 2: 4x4x4 for cols 4-7
    # B 2x2 blocks (cols 4-7 of 4x9):
    B00_p2 = [4, 5, 13, 14]   # (0,4),(0,5),(1,4),(1,5) -> 0*9+4, 0*9+5, 1*9+4, 1*9+5
    B01_p2 = [6, 7, 15, 16]   # (0,6),(0,7),(1,6),(1,7)
    B10_p2 = [22, 23, 31, 32] # (2,4),(2,5),(3,4),(3,5)
    B11_p2 = [24, 25, 33, 34] # (2,6),(2,7),(3,6),(3,7)
    
    # C 2x2 blocks (cols 4-7 of C):
    C00_p2 = [4, 5, 8, 9]     # (k=4? No, C is 4x4 with k=0..3. Wait, C has only 4 columns.
    # Oops! C is 4x4 (p*n = 9*4 = 36 elements? No, C is 4x4 = 16 elements? 
    # Wait, the problem says 4x4 * 4x9 -> 4x9. So C is 4x9, not 4x4!
    # I made an error. C is 4x9, with trace convention C[k,i] at k*N+i where N=4.
    # So C has 9*4=36 elements.
    
    # Let me restart with correct dimensions.
    # A: 4x4 = 16 elements
    # B: 4x9 = 36 elements  
    # C: 4x9 = 36 elements (trace convention: C[k,i] at k*4+i)
    
    # DIM_U = 16, DIM_V = 36, DIM_W = 36
    
    # For col k of C, the element C[k,i] is at k*4+i.
    # For cols 0-3: C indices 0..15
    # For cols 4-7: C indices 16..31
    # For col 8: C indices 32..35
    
    # Let me redo this properly.
    pass
    
    # I'll rebuild from scratch with correct understanding.
    
    # Start over
    U = [[0] * rank for _ in range(DIM_U)]
    V = [[0] * rank for _ in range(DIM_V)]
    W = [[0] * rank for _ in range(DIM_W)]
    col = 0
    
    def _add_222_strassen(U, V, W, col, a_idx, b_idx, c_idx):
        for (u, v, w) in _STRASSEN_222:
            for t in range(4):
                if u[t]:
                    U[a_idx[t]][col] += u[t]
                if v[t]:
                    V[b_idx[t]][col] += v[t]
                if w[t]:
                    W[c_idx[t]][col] += w[t]
            col += 1
        return col
    
    # A 2x2 blocks (4x4 A, row-major):
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # For each column group, define B and C 2x2 blocks.
    
    # Col group 0-3 (k=0,1,2,3):
    # B 2x2 blocks:
    B_g0_00 = [0, 1, 9, 10]    # j=0,1; k=0,1
    B_g0_01 = [2, 3, 11, 12]   # j=0,1; k=2,3
    B_g0_10 = [18, 19, 27, 28] # j=2,3; k=0,1
    B_g0_11 = [20, 21, 29, 30] # j=2,3; k=2,3
    # C 2x2 blocks (C[k,i] at k*4+i, k=0..3, i=0..3):
    C_g0_00 = [0, 1, 4, 5]     # k=0,1; i=0,1
    C_g0_01 = [2, 3, 6, 7]     # k=0,1; i=2,3
    C_g0_10 = [8, 9, 12, 13]   # k=2,3; i=0,1
    C_g0_11 = [10, 11, 14, 15] # k=2,3; i=2,3
    
    products_g0 = [
        (A00, B_g0_00, C_g0_00),
        (A01, B_g0_10, C_g0_00),
        (A00, B_g0_01, C_g0_01),
        (A01, B_g0_11, C_g0_01),
        (A10, B_g0_00, C_g0_10),
        (A11, B_g0_10, C_g0_10),
        (A10, B_g0_01, C_g0_11),
        (A11, B_g0_11, C_g0_11),
    ]
    
    for a, b, c in products_g0:
        col = _add_222_strassen(U, V, W, col, a, b, c)
    
    # Col group 4-7 (k=4,5,6,7):
    B_g1_00 = [4, 5, 13, 14]
    B_g1_01 = [6, 7, 15, 16]
    B_g1_10 = [22, 23, 31, 32]
    B_g1_11 = [24, 25, 33, 34]
    C_g1_00 = [16, 17, 20, 21]    # k=4,5; i=0,1 -> 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21
    C_g1_01 = [18, 19, 22, 23]    # k=4,5; i=2,3
    C_g1_10 = [24, 25, 28, 29]    # k=6,7; i=0,1
    C_g1_11 = [26, 27, 30, 31]    # k=6,7; i=2,3
    
    products_g1 = [
        (A00, B_g1_00, C_g1_00),
        (A01, B_g1_10, C_g1_00),
        (A00, B_g1_01, C_g1_01),
        (A01, B_g1_11, C_g1_01),
        (A10, B_g1_00, C_g1_10),
        (A11, B_g1_10, C_g1_10),
        (A10, B_g1_01, C_g1_11),
        (A11, B_g1_11, C_g1_11),
    ]
    
    for a, b, c in products_g1:
        col = _add_222_strassen(U, V, W, col, a, b, c)
    
    # Col group 8 (k=8):
    # B 2x2 blocks: only one column (k=8), so B blocks are 2x1, not 2x2.
    # I need to handle this differently. For k=8, it's a 4x4 * 4x1 -> 4x1 product.
    # Schoolbook: 4*4*1 = 16 terms.
    
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U[i*4 + j][col] += 1
                V[j*9 + 8][col] += 1
                W[8*4 + i][col] += 1
                col += 1
    
    # col should be 56 + 56 + 16 = 128
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    
    # Verify
    if time.time() < deadline:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        ok = True
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    a = i*4 + j
                    b = j*9 + k
                    c = k*4 + i
                    total = 0.0
                    for r in range(rank):
                        total += Ua[a, r] * Va[b, r] * Wa[c, r]
                    if abs(total - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 144
    rank = 144
    U = [[0] * rank for _ in range(DIM_U)]
    V = [[0] * rank for _ in range(DIM_V)]
    W = [[0] * rank for _ in range(DIM_W)]
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                U[i*4 + j][c] = 1
                V[j*9 + k][c] = 1
                W[k*4 + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
