import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # We will build the decomposition for 4x4x4 and 4x4x2 separately and concatenate
    
    # Helper to add a Strassen 2x2x2 decomposition
    # This is a bit complex to do generically with indices.
    # It's easier to hardcode the Strassen formulas for the blocks.
    
    # Let's use a library-like approach or just hardcode the 49+28 terms.
    # Given the complexity, I will write a function that generates the terms for a 2x2x2 Strassen block
    # given the global indices of A, B, C.
    
    # A is 4x4. Indices: i (0-3), j (0-3). Global A index: i*4+j.
    # B is 4x6. Indices: j (0-3), k (0-5). Global B index: j*6+k.
    # C is 6x4. Indices: k (0-5), i (0-3). Global C index: k*4+i. (Task says W[k*n+i,r])
    
    rank = 77
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    col = 0
    
    # Part 1: 4x4x4 (Left 4 cols of B)
    # B_L is 4x4. j in 0-3, k in 0-3.
    # We use recursive Strassen for 4x4x4.
    # This requires 49 terms.
    # I will implement a recursive Strassen generator.
    
    # Part 2: 4x4x2 (Right 2 cols of B)
    # B_R is 4x2. j in 0-3, k in 4-5.
    # We use Block Strassen: 4 blocks of 2x2x2.
    # A is split into 2x2 blocks.
    # B_R is split into 2x2 blocks (top and bottom).
    
    # Since implementing recursive Strassen generically is error-prone, 
    # and I have time, I will try to construct the 2x2x2 Strassen terms manually 
    # for the 4x4x2 part, and use a known 49-term decomposition for 4x4x4.
    
    # Actually, I can just use the 4x4x4 Strassen decomposition and ignore the last 2 columns?
    # No, that would give C = A B_L. I need A B_R too.
    # So I need a separate decomposition for A B_R.
    
    # Let's define a function to add a 2x2x2 Strassen term
    def add_strassen_2x2x2(a_offsets, b_offsets, c_offsets):
        # a_offsets: dict mapping 'a','b','c','d' to global A indices
        # b_offsets: dict mapping 'e','f','g','h' to global B indices
        # c_offsets: dict mapping 'c11','c12','c21','c22' to global C indices
        nonlocal col
        # M1 = (a+d)(e+h)
        U[a_offsets['a']][col] = 1
        U[a_offsets['d']][col] = 1
        V[b_offsets['e']][col] = 1
        V[b_offsets['h']][col] = 1
        W[c_offsets['c11']][col] = 1
        W[c_offsets['c22']][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[a_offsets['c']][col] = 1
        U[a_offsets['d']][col] = 1
        V[b_offsets['e']][col] = 1
        W[c_offsets['c21']][col] = 1
        W[c_offsets['c22']][col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[a_offsets['a']][col] = 1
        V[b_offsets['f']][col] = 1
        V[b_offsets['h']][col] = -1
        W[c_offsets['c12']][col] = 1
        W[c_offsets['c22']][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[a_offsets['d']][col] = 1
        V[b_offsets['g']][col] = 1
        V[b_offsets['e']][col] = -1
        W[c_offsets['c11']][col] = -1
        W[c_offsets['c21']][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[a_offsets['a']][col] = 1
        U[a_offsets['b']][col] = 1
        V[b_offsets['h']][col] = 1
        W[c_offsets['c11']][col] = 1
        W[c_offsets['c12']][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[a_offsets['c']][col] = 1
        U[a_offsets['a']][col] = -1
        V[b_offsets['e']][col] = 1
        V[b_offsets['f']][col] = 1
        W[c_offsets['c22']][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[a_offsets['b']][col] = 1
        U[a_offsets['d']][col] = -1
        V[b_offsets['g']][col] = 1
        V[b_offsets['h']][col] = 1
        W[c_offsets['c11']][col] = -1
        W[c_offsets['c22']][col] = -1
        col += 1

    # Part 2: 4x4x2 (Right 2 columns of B, k=4,5)
    # Split A into 4 blocks 2x2:
    # A11: i=0,1; j=0,1
    # A12: i=0,1; j=2,3
    # A21: i=2,3; j=0,1
    # A22: i=2,3; j=2,3
    
    # Split B_R into 2 blocks 2x2:
    # B_R1: j=0,1; k=4,5
    # B_R2: j=2,3; k=4,5
    
    # Product C_R is 4x2:
    # C_R1: i=0,1; k=4,5 (Top-Left of C_R, but C is 6x4, so i is row of A, k is col of B)
    # Wait, C index is k*n + i.
    # C_R corresponds to k=4,5.
    # C_R1: i=0,1; k=4,5. Indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21.
    # C_R2: i=2,3; k=4,5. Indices: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23.
    
    # Block 1: A11 * B_R1 -> C_R1 (Top part)
    a_off = {'a': 0*4+0, 'b': 0*4+1, 'c': 1*4+0, 'd': 1*4+1}
    b_off = {'e': 0*6+4, 'f': 0*6+5, 'g': 1*6+4, 'h': 1*6+5}
    c_off = {'c11': 16, 'c12': 17, 'c21': 20, 'c22': 21} # c11 is i=0,k=4? No.
    # C index k*n+i.
    # c11 of result 2x2 corresponds to i=0, k=4?
    # Result of A11(2x2) * B_R1(2x2) is 2x2.
    # Rows of result correspond to i=0,1. Cols correspond to k=4,5.
    # So (0,0) of result is i=0, k=4. Index 4*4+0 = 16.
    # (0,1) is i=0, k=5. Index 5*4+0 = 20? No.
    # W index is k*n+i.
    # k=4, i=0 -> 16.
    # k=5, i=0 -> 20.
    # k=4, i=1 -> 17.
    # k=5, i=1 -> 21.
    # So c11 (row 0, col 0 of local 2x2) is i=0, k=4 -> 16.
    # c12 (row 0, col 1) is i=0, k=5 -> 20.
    # c21 (row 1, col 0) is i=1, k=4 -> 17.
    # c22 (row 1, col 1) is i=1, k=5 -> 21.
    
    c_off = {'c11': 16, 'c12': 20, 'c21': 17, 'c22': 21}
    add_strassen_2x2x2(a_off, b_off, c_off)
    
    # Block 2: A12 * B_R2 -> C_R1 (Top part)
    # A12: i=0,1; j=2,3
    a_off = {'a': 0*4+2, 'b': 0*4+3, 'c': 1*4+2, 'd': 1*4+3}
    # B_R2: j=2,3; k=4,5
    b_off = {'e': 2*6+4, 'f': 2*6+5, 'g': 3*6+4, 'h': 3*6+5}
    # Result adds to C_R1 (i=0,1; k=4,5)
    c_off = {'c11': 16, 'c12': 20, 'c21': 17, 'c22': 21}
    add_strassen_2x2x2(a_off, b_off, c_off)
    
    # Block 3: A21 * B_R1 -> C_R2 (Bottom part)
    # A21: i=2,3; j=0,1
    a_off = {'a': 2*4+0, 'b': 2*4+1, 'c': 3*4+0, 'd': 3*4+1}
    # B_R1: j=0,1; k=4,5
    b_off = {'e': 0*6+4, 'f': 0*6+5, 'g': 1*6+4, 'h': 1*6+5}
    # Result is i=2,3; k=4,5.
    # i=2, k=4 -> 18. i=2, k=5 -> 22.
    # i=3, k=4 -> 19. i=3, k=5 -> 23.
    c_off = {'c11': 18, 'c12': 22, 'c21': 19, 'c22': 23}
    add_strassen_2x2x2(a_off, b_off, c_off)
    
    # Block 4: A22 * B_R2 -> C_R2 (Bottom part)
    # A22: i=2,3; j=2,3
    a_off = {'a': 2*4+2, 'b': 2*4+3, 'c': 3*4+2, 'd': 3*4+3}
    # B_R2: j=2,3; k=4,5
    b_off = {'e': 2*6+4, 'f': 2*6+5, 'g': 3*6+4, 'h': 3*6+5}
    # Result is i=2,3; k=4,5.
    c_off = {'c11': 18, 'c12': 22, 'c21': 19, 'c22': 23}
    add_strassen_2x2x2(a_off, b_off, c_off)
    
    # Now Part 1: 4x4x4 (Left 4 cols of B, k=0..3)
    # This is a full 4x4x4 Strassen.
    # I need to generate the 49 terms.
    # I will implement a recursive Strassen for 4x4x4.
    
    def add_strassen_4x4x4(a_base, b_base, c_base):
        # a_base: (i0, j0) top-left of A block
        # b_base: (j0, k0) top-left of B block
        # c_base: (k0, i0) top-left of C block
        nonlocal col
        # Split into 2x2 blocks
        # A11, A12, A21, A22
        # B1, B2 (top/bottom 2x4? No, B is 4x4, so B1 is 2x4, B2 is 2x4)
        # Wait, Strassen for 4x4x4 splits A into 2x2, B into 2x4?
        # No, standard Strassen splits both into 2x2.
        # A is 4x4. B is 4x4.
        # A11(2x2) A12(2x2)
        # A21(2x2) A22(2x2)
        # B11(2x2) B12(2x2)
        # B21(2x2) B22(2x2)
        
        # M1 = (A11+A22)(B11+B22)
        # This requires decomposing (A11+A22) and (B11+B22).
        # This is getting too complex to hardcode.
        
        # Alternative: Use the fact that 4x4x4 Strassen rank is 49.
        # I can just use the 28 terms from 4x4x2 logic for the first 2 cols,
        # and another 28 terms for the next 2 cols?
        # 4x4x4 = 4x4x2 + 4x4x2.
        # Rank 28 + 28 = 56.
        # But Strassen 4x4x4 is 49.
        # 56 > 49. So I should use the 49 term decomposition.
        
        # Since I don't have the 49 term formulas memorized, I will use the 56 term decomposition (two 4x4x2 blocks) for the first 4 columns as well.
        # Total rank: 28 (for k=0,1) + 28 (for k=2,3) + 28 (for k=4,5) = 84.
        # 84 < 96.
        # Is 84 better than 77? No.
        # But 84 is easier to implement correctly.
        # Let's stick to 84 if I can't get 49.
        # Wait, I can implement the 4x4x2 block logic for k=0,1 and k=2,3.
        
        # Let's do that.
        # Block k=0,1
        # B_L1: j=0..3, k=0,1
        # A11*B_L1_top + A12*B_L1_bot ...
        
        # This is just repeating the 4x4x2 logic for different k ranges.
        
        pass

    # Let's refine the 84 rank strategy.
    # It's structurally simpler.
    # I will reset U,V,W and col.
    U = [[0]*84 for _ in range(16)]
    V = [[0]*84 for _ in range(24)]
    W = [[0]*84 for _ in range(24)]
    col = 0
    
    def add_4x4x2_block(k_start):
        # Adds decomposition for A * B[:, k_start:k_start+2]
        nonlocal col
        # B block is 4x2. j=0..3, k=k_start, k_start+1.
        # Split A into 4 2x2 blocks.
        # Split B block into 2 2x2 blocks (top j=0,1 and bottom j=2,3).
        
        # A11: i=0,1; j=0,1
        a11 = {'a': 0, 'b': 1, 'c': 4, 'd': 5}
        # A12: i=0,1; j=2,3
        a12 = {'a': 2, 'b': 3, 'c': 6, 'd': 7}
        # A21: i=2,3; j=0,1
        a21 = {'a': 8, 'b': 9, 'c': 12, 'd': 13}
        # A22: i=2,3; j=2,3
        a22 = {'a': 10, 'b': 11, 'c': 14, 'd': 15}
        
        # B_top: j=0,1; k=k_start, k_start+1
        b_top = {'e': 0*6+k_start, 'f': 0*6+k_start+1, 'g': 1*6+k_start, 'h': 1*6+k_start+1}
        # B_bot: j=2,3; k=k_start, k_start+1
        b_bot = {'e': 2*6+k_start, 'f': 2*6+k_start+1, 'g': 3*6+k_start, 'h': 3*6+k_start+1}
        
        # C_top: i=0,1; k=k_start, k_start+1
        c_top = {'c11': k_start*4+0, 'c12': (k_start+1)*4+0, 'c21': k_start*4+1, 'c22': (k_start+1)*4+1}
        # C_bot: i=2,3; k=k_start, k_start+1
        c_bot = {'c11': k_start*4+2, 'c12': (k_start+1)*4+2, 'c21': k_start*4+3, 'c22': (k_start+1)*4+3}
        
        # 4 products
        add_strassen_2x2x2(a11, b_top, c_top)
        add_strassen_2x2x2(a12, b_bot, c_top)
        add_strassen_2x2x2(a21, b_top, c_bot)
        add_strassen_2x2x2(a22, b_bot, c_bot)
        
    add_4x4x2_block(0)
    add_4x4x2_block(2)
    add_4x4x2_block(4)
    
    return {"U": U, "V": V, "W": W}
