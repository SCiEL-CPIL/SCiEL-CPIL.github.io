import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = 78
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k, col_start, block_type):
        """
        Adds a decomposition block.
        a_off_i, a_off_j: top-left of A (n x m)
        b_off_j, b_off_k: top-left of B (m x p)
        c_off_i, c_off_k: top-left of C (n x p)
        col_start: starting column index
        block_type: 'strassen222' or 'schoolbook221' or 'schoolbook147'
        """
        col = col_start
        
        if block_type == 'schoolbook147':
            # 1x4x7 block
            # A: 1 row, 4 cols. B: 4 rows, 7 cols. C: 1 row, 7 cols.
            for j in range(4):
                for k in range(7):
                    u_idx = (a_off_i + 0) * m + (a_off_j + j)
                    v_idx = (b_off_j + j) * p + (b_off_k + k)
                    w_idx = (c_off_k + k) * n + (c_off_i + 0)
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
            
        elif block_type == 'strassen222':
            # 2x2x2 Strassen
            # Local indices 0,1 for A, B, C
            
            # M1: (A00+A11)(B00+B11)
            # A00, A11 -> +1
            # B00, B11 -> +1
            # C00 -> +1, C11 -> +1, C10 -> -1? 
            # Let's use the reconstruction formulas to set W.
            
            # M1
            U[(a_off_i+0)*m + (a_off_j+0)][col] += 1
            U[(a_off_i+1)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+1)][col] += 1
            W[(c_off_k+0)*n + (c_off_i+0)][col] += 1 # C00
            W[(c_off_k+1)*n + (c_off_i+1)][col] += 1 # C11
            # Wait, M1 contributes to C00 (+1) and C11 (+1) in the standard formula?
            # C00 = M1 + M4 - M5 + M7
            # C11 = M1 - M2 + M3 + M6
            # So M1 contributes +1 to C00 and +1 to C11.
            col += 1
            
            # M2: (A10+A11)B00
            U[(a_off_i+1)*m + (a_off_j+0)][col] += 1
            U[(a_off_i+1)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] += 1
            # C10 = M2 + M4 -> M2 contributes +1 to C10
            W[(c_off_k+0)*n + (c_off_i+1)][col] += 1 # C10
            # C11 = M1 - M2 ... -> M2 contributes -1 to C11
            W[(c_off_k+1)*n + (c_off_i+1)][col] -= 1 # C11
            col += 1
            
            # M3: A00(B01-B11)
            U[(a_off_i+0)*m + (a_off_j+0)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+1)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+1)][col] -= 1
            # C01 = M3 + M5 -> M3 contributes +1 to C01
            W[(c_off_k+1)*n + (c_off_i+0)][col] += 1 # C01
            # C11 = ... + M3 ... -> M3 contributes +1 to C11
            W[(c_off_k+1)*n + (c_off_i+1)][col] += 1 # C11
            col += 1
            
            # M4: A11(B10-B00)
            U[(a_off_i+1)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+0)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] -= 1
            # C00 = ... + M4 ... -> M4 contributes +1 to C00
            W[(c_off_k+0)*n + (c_off_i+0)][col] += 1 # C00
            # C10 = M2 + M4 -> M4 contributes +1 to C10
            W[(c_off_k+0)*n + (c_off_i+1)][col] += 1 # C10
            col += 1
            
            # M5: (A00+A01)B11
            U[(a_off_i+0)*m + (a_off_j+0)][col] += 1
            U[(a_off_i+0)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+1)][col] += 1
            # C00 = ... - M5 ... -> M5 contributes -1 to C00
            W[(c_off_k+0)*n + (c_off_i+0)][col] -= 1 # C00
            # C01 = M3 + M5 -> M5 contributes +1 to C01
            W[(c_off_k+1)*n + (c_off_i+0)][col] += 1 # C01
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[(a_off_i+1)*m + (a_off_j+0)][col] += 1
            U[(a_off_i+0)*m + (a_off_j+0)][col] -= 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+1)][col] += 1
            # C11 = ... + M6 ... -> M6 contributes +1 to C11
            W[(c_off_k+1)*n + (c_off_i+1)][col] += 1 # C11
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[(a_off_i+0)*m + (a_off_j+1)][col] += 1
            U[(a_off_i+1)*m + (a_off_j+1)][col] -= 1
            V[(b_off_j+1)*p + (b_off_k+0)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+1)][col] += 1
            # C00 = ... + M7 ... -> M7 contributes +1 to C00
            W[(c_off_k+0)*n + (c_off_i+0)][col] += 1 # C00
            col += 1
            
        elif block_type == 'schoolbook221':
            # 2x2x1 block
            # A: 2x2, B: 2x1, C: 2x1
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            
            # Term 1: A00 * B0 -> C0
            U[(a_off_i+0)*m + (a_off_j+0)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] += 1
            W[(c_off_k+0)*n + (c_off_i+0)][col] += 1 # C0
            col += 1
            
            # Term 2: A01 * B1 -> C0
            U[(a_off_i+0)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+0)][col] += 1
            W[(c_off_k+0)*n + (c_off_i+0)][col] += 1 # C0
            col += 1
            
            # Term 3: A10 * B0 -> C1
            U[(a_off_i+1)*m + (a_off_j+0)][col] += 1
            V[(b_off_j+0)*p + (b_off_k+0)][col] += 1
            W[(c_off_k+0)*n + (c_off_i+1)][col] += 1 # C1
            col += 1
            
            # Term 4: A11 * B1 -> C1
            U[(a_off_i+1)*m + (a_off_j+1)][col] += 1
            V[(b_off_j+1)*p + (b_off_k+0)][col] += 1
            W[(c_off_k+0)*n + (c_off_i+1)][col] += 1 # C1
            col += 1

    # 1. Part 2: 1x4x7 (Bottom row of A/C)
    # A offset: i=2, j=0. B offset: j=0, k=0. C offset: i=2, k=0.
    # Size: A 1x4, B 4x7, C 1x7.
    add_block(2, 0, 0, 0, 2, 0, 0, 'schoolbook147')
    # Ends at col 28.

    # 2. Part 1a: 2x4x7 (Top 2 rows of A/C, Left 2 cols of A, Top 2 rows of B)
    # Split B into 2x2, 2x2, 2x2, 2x1 blocks.
    
    # Sub-block 0: 2x2x2 (B cols 0,1)
    # A offset: i=0, j=0. B offset: j=0, k=0. C offset: i=0, k=0.
    add_block(0, 0, 0, 0, 0, 0, 28, 'strassen222')
    # Ends at col 35.
    
    # Sub-block 1: 2x2x2 (B cols 2,3)
    # A offset: i=0, j=0. B offset: j=0, k=2. C offset: i=0, k=2.
    add_block(0, 0, 0, 2, 0, 2, 35, 'strassen222')
    # Ends at col 42.
    
    # Sub-block 2: 2x2x2 (B cols 4,5)
    # A offset: i=0, j=0. B offset: j=0, k=4. C offset: i=0, k=4.
    add_block(0, 0, 0, 4, 0, 4, 42, 'strassen222')
    # Ends at col 49.
    
    # Sub-block 3: 2x2x1 (B col 6)
    # A offset: i=0, j=0. B offset: j=0, k=6. C offset: i=0, k=6.
    add_block(0, 0, 0, 6, 0, 6, 49, 'schoolbook221')
    # Ends at col 53.
    
    # 3. Part 1b: 2x4x7 (Top 2 rows of A/C, Right 2 cols of A, Bottom 2 rows of B)
    
    # Sub-block 0: 2x2x2 (B cols 0,1)
    # A offset: i=0, j=2. B offset: j=2, k=0. C offset: i=0, k=0.
    add_block(0, 2, 2, 0, 0, 0, 53, 'strassen222')
    # Ends at col 60.
    
    # Sub-block 1: 2x2x2 (B cols 2,3)
    # A offset: i=0, j=2. B offset: j=2, k=2. C offset: i=0, k=2.
    add_block(0, 2, 2, 2, 0, 2, 60, 'strassen222')
    # Ends at col 67.
    
    # Sub-block 2: 2x2x2 (B cols 4,5)
    # A offset: i=0, j=2. B offset: j=2, k=4. C offset: i=0, k=4.
    add_block(0, 2, 2, 4, 0, 4, 67, 'strassen222')
    # Ends at col 74.
    
    # Sub-block 3: 2x2x1 (B col 6)
    # A offset: i=0, j=2. B offset: j=2, k=6. C offset: i=0, k=6.
    add_block(0, 2, 2, 6, 0, 6, 74, 'schoolbook221')
    # Ends at col 78.

    return {"U": U, "V": V, "W": W}
