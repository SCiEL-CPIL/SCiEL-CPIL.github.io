"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U, V, W are lists of lists of size (4, 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: (A10+A11)B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: A00(B01-B11) -> C01
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: A11(B10-B00) -> C10, C11
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # Term 5: (A00+A01)B11 -> C01, C11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1 # Wait, C01 gets +P5, C11 gets -P5?
    # Strassen: C01 = P3 + P5. C11 = P1 - P2 + P3 + P6.
    # My Term 5 is P5.
    # C01: +1. C11: 0?
    # Let's re-verify Strassen formulas.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # My Term 5: P5.
    # W[1][4] (C01) should be 1.
    # W[3][4] (C11) should be 0.
    # I put -1 above. That was wrong.
    W[3][4] = 0
    
    # Term 6: (A10-A00)(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    U, V, W are lists of lists of size (16, 49).
    """
    U2, V2, W2 = _strassen_2()
    # U2, V2, W2 are 4x7
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Strassen 4x4x4 uses 7 terms, each is a 2x2x2 multiplication.
    # We map the 7 terms to column offsets 0..6, 7..13, etc.
    
    # Define the 7 combinations of 2x2 blocks for A, B
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # Indices in 4x4 A:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks: B00, B01, B10, B11
    # Indices in 4x4 B:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C blocks: C00, C01, C10, C11
    # Indices in 4x4 C:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Helper to add a 2x2x2 decomposition
    def add_term(u_offset, v_offset, w_offset, col_offset):
        # u_offset: start index of A block in U (0, 8)
        # v_offset: start index of B block in V (0, 8)
        # w_offset: start index of C block in W (0, 8)
        # col_offset: start column in U,V,W (0, 7, 14...)
        
        for r in range(4):
            for c in range(7):
                # Map local indices 0..3 to global indices
                # Local 0,1 -> row u_offset, u_offset+1?
                # No, U2 is 4x7. Rows 0,1,2,3 correspond to A block entries.
                # We need to map A block entries to global A indices.
                
                # This is getting complex to map manually.
                # Alternative: Construct U, V, W for 4x4x4 by embedding 2x2x2.
                
                pass

    # Easier approach:
    # U is 16x49.
    # Each of the 7 Strassen terms corresponds to a 2x2x2 multiplication.
    # We can define the coefficients for the 7 terms on the 4x4 blocks.
    
    # Let's just use the known Strassen coefficients for 4x4x4 directly.
    # Or, better, construct U, V, W by iterating over the 7 terms and adding the 2x2x2 pattern.
    
    # Block indices in 4x4 matrix (row-major):
    # Block (bi, bj) where bi, bj in {0,1}
    # Start index = bi*8 + bj*2
    # Entries: start, start+1, start+4, start+5
    
    def get_block_indices(bi, bj):
        start = bi * 8 + bj * 2
        return [start, start+1, start+4, start+5]
    
    # Strassen terms for 4x4x4:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B01-B11)
    # 4. A11(B10-B00)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    
    terms = [
        # (u_blocks, v_blocks, w_blocks)
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # P1
        ([(2,1), (3,1)], [(0,1)], [(2,1)]), # P2 -> C10
        ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # P3 -> C01
        ([(3,1)], [(2,1), (0,-1)], [(2,1), (3,1)]), # P4 -> C10, C11
        ([(0,1), (1,1)], [(3,1)], [(1,1)]), # P5 -> C01
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # P6 -> C11
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]), # P7 -> C00
    ]
    
    # Wait, P4 contributes to C10 and C11?
    # C10 = P2 + P4.
    # C11 = P1 - P2 + P3 + P6.
    # So P4 only contributes to C10.
    # My term 4 above has w_blocks [(2,1), (3,1)]. That means C10 and C11.
    # Let's check P4 = A11(B10-B00).
    # C10 = A00B10 + A01B11 + A10B00 + A11B01? No.
    # C10 (block) = A00 B10 + A01 B11? No.
    # C = [[C00, C01], [C10, C11]].
    # C10 = A00 B10 + A01 B11? No.
    # C10 is bottom-left.
    # C10 = A00 B10 + A01 B11 is wrong.
    # C10 = A00 B10 + A01 B11 is for C11?
    # C = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]]
    # So C10 = A10B00 + A11B10.
    # P2 = (A10+A11)B00 = A10B00 + A11B00.
    # P4 = A11(B10-B00) = A11B10 - A11B00.
    # P2 + P4 = A10B00 + A11B10. Correct.
    # So P4 contributes to C10.
    # Does P4 contribute to C11? No.
    # So w_blocks for P4 should be [(2,1)] (C10).
    
    # Let's re-verify all w_blocks.
    # P1: C00, C11. w: [(0,1), (3,1)]
    # P2: C10. w: [(2,1)]
    # P3: C01. w: [(1,1)]
    # P4: C10. w: [(2,1)]
    # P5: C01. w: [(1,1)]
    # P6: C11. w: [(3,1)]
    # P7: C00. w: [(0,1)]
    
    # Corrected terms:
    terms = [
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # P1
        ([(2,1), (3,1)], [(0,1)], [(2,1)]), # P2
        ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # P3
        ([(3,1)], [(2,1), (0,-1)], [(2,1)]), # P4
        ([(0,1), (1,1)], [(3,1)], [(1,1)]), # P5
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # P6
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]), # P7
    ]
    
    for t_idx, (u_blocks, v_blocks, w_blocks) in enumerate(terms):
        col_start = t_idx * 7
        
        # For each block in u_blocks, add the 2x2x2 pattern
        # We need to sum the contributions from all blocks in a term.
        # Since the 2x2x2 decomposition is linear, we can just add the U,V,W patterns.
        
        # Initialize local U, V, W for this term (16x7, 16x7, 16x7)
        local_U = [[0]*7 for _ in range(16)]
        local_V = [[0]*7 for _ in range(16)]
        local_W = [[0]*7 for _ in range(16)]
        
        # Add contributions from each block
        for (bi, coeff) in u_blocks:
            inds = get_block_indices(bi // 2, bi % 2) # bi: 0=A00, 1=A01, 2=A10, 3=A11
            # Map local 2x2 indices (0,1,2,3) to global indices
            # Local 0 -> inds[0], 1 -> inds[1], 2 -> inds[2], 3 -> inds[3]
            for r in range(4):
                for c in range(7):
                    local_U[inds[r]][c] += coeff * U2[r][c]
        
        for (bi, coeff) in v_blocks:
            inds = get_block_indices(bi // 2, bi % 2)
            for r in range(4):
                for c in range(7):
                    local_V[inds[r]][c] += coeff * V2[r][c]
        
        for (bi, coeff) in w_blocks:
            inds = get_block_indices(bi // 2, bi % 2)
            for r in range(4):
                for c in range(7):
                    local_W[inds[r]][c] += coeff * W2[r][c]
        
        # Add local_U, local_V, local_W to global U, V, W
        for r in range(16):
            for c in range(7):
                U[r][col_start + c] += local_U[r][c]
                V[r][col_start + c] += local_V[r][c]
                W[r][col_start + c] += local_W[r][c]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    U4, V4, W4 = _strassen_4()
    
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    
    # Construct global U (16x98), V (32x98), W (32x98)
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Part 1: B columns 0-3
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            V[r][c] = V4[r][c]
            W[r][c] = W4[r][c]
            
    # Part 2: B columns 4-7
    # U is same
    for r in range(16):
        for c in range(49):
            U[r][c + 49] = U4[r][c]
            
    # V rows 16-31 correspond to B2
    for r in range(16):
        for c in range(49):
            V[r + 16][c + 49] = V4[r][c]
            
    # W rows 16-31 correspond to C2
    for r in range(16):
        for c in range(49):
            W[r + 16][c + 49] = W4[r][c]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
